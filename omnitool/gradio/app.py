#!/usr/bin/env python
"""
Run with:
python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
"""

import os
import json
from pathlib import Path
from datetime import datetime
from enum import StrEnum
from functools import partial
from typing import cast
import argparse
import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from loop import (
    APIProvider,
    sampling_loop_sync,
)
from tools import ToolResult
import requests
from requests.exceptions import RequestException
import base64

# =============================================================================
# Config and directories
# =============================================================================
CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"
DATA_DIR = Path(__file__).parent / "data"

INTRO_TEXT = '''
OmniParser lets you turn any vision-language model into an AI agent. We currently support **OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use (Sonnet).**

Type a message and press submit to start OmniTool. Press stop to pause, and press the trash icon in the chat to clear the message history.
'''

# =============================================================================
# Command-line arguments
# =============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument(
        "--windows_host_url",
        type=str,
        default="localhost:8006",
        help="VNC UI host:port (e.g. localhost:8006)",
    )
    parser.add_argument(
        "--omniparser_server_url",
        type=str,
        default="localhost:8000",
        help="OmniParser FastAPI server host:port",
    )
    return parser.parse_args()

args = parse_arguments()

# =============================================================================
# JSON Task Runner utilities
# =============================================================================
def load_json_files() -> list[str]:
    """Return list of .json files in data directory."""
    return [p.name for p in DATA_DIR.glob("*.json") if p.is_file()]

def process_json_input(user_input, state):
    # validation 없이 바로 메시지 보내고 실행
    state["messages"].append({
        "role": Sender.USER,
        "content": [TextBlock(type="text", text=user_input)]
    })
    state["chatbot_messages"].append((user_input, None))
    yield state["chatbot_messages"]
    for _ in sampling_loop_sync(
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state["chatbot_messages"], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        max_tokens=16384,
        omniparser_url=args.omniparser_server_url
    ):
        yield state["chatbot_messages"]

def run_json_tasks(selected_file: str, state):
    """Sequentially execute 'data' or 'task' fields from selected JSON via process_input."""
    # Populate API key from saved keys
    state["api_key"] = state.get(f"{state['provider']}_api_key", "")
    # Reset state for fresh run
    state["skip_validation"] = True
    state.update({
        "stop": False,
        "messages": [],
        "responses": {},
        "tools": {},
        "chatbot_messages": [],
    })

    tasks = json.load(open(DATA_DIR / selected_file, encoding="utf-8"))
    for item in tasks:
        # support both 'data' and 'task' keys
        content = item.get("data") or item.get("task")
        if not content:
            continue
        for _ in process_json_input(content, state):
            yield state["chatbot_messages"]
    state["skip_validation"] = False
    return state["chatbot_messages"]

# =============================================================================
# State, authentication, and storage
# =============================================================================
class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state(state):
    """Initialize the persistent state dict."""
    defaults = {
        "messages": [],
        "model": "omniparser + gpt-4o",
        "provider": "openai",
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "api_key": "",
        "auth_validated": False,
        "responses": {},
        "tools": {},
        "only_n_most_recent_images": 2,
        "chatbot_messages": [],
        "stop": False,
    }
    for k,v in defaults.items():
        state.setdefault(k, v)


async def main(state):
    setup_state(state)
    return "Setup completed"


def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC and not api_key:
        return "Enter your Anthropic API key to continue."
    if provider == APIProvider.BEDROCK:
        import boto3
        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "Your Google Cloud credentials are not set up correctly."


def load_from_storage(filename: str) -> str | None:
    try:
        fp = CONFIG_DIR / filename
        if fp.exists():
            data = fp.read_text().strip()
            return data or None
    except:
        pass
    return None

def save_to_storage(filename: str, data: str) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        fp = CONFIG_DIR / filename
        fp.write_text(data)
        fp.chmod(0o600)
    except:
        pass

# =============================================================================
# Callbacks for API responses, tools, and chat rendering
# =============================================================================

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_state[datetime.now().isoformat()] = response


def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output



def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render(msg):
        if isinstance(msg, str):
            return msg
        if isinstance(msg, ToolResult):
            if msg.output:
                return msg.output
            if msg.error:
                return f"Error: {msg.error}"
            if msg.base64_image:
                return f'<img src="data:image/png;base64,{msg.base64_image}">'
        if isinstance(msg, BetaTextBlock) or isinstance(msg, TextBlock):
            return f"Analysis: {msg.text}"
        if isinstance(msg, BetaToolUseBlock) or isinstance(msg, ToolUseBlock):
            return f"Next I will perform action: {msg.input}"
        return msg

    out = _render(message)
    if sender == "bot":
        chatbot_state.append((None, out))
    else:
        chatbot_state.append((out, None))

# =============================================================================
# Validation before running tasks or chat
# =============================================================================

def valid_params(user_input, state):
    """
    Probe:
     - Windows control API (host_ip:5000)
     - OmniParser server
    """
    errors = []

    probes = [
        ("Windows Host (control API)", args.windows_host_url),
        ("OmniParser Server",    args.omniparser_server_url),
    ]
    for name, host_url in probes:
        url = f"http://{host_url}/probe"
        try:
            r = requests.get(url, timeout=3)
            if r.status_code != 200:
                errors.append(f"{name} ({url}) not responding")
        except RequestException:
            errors.append(f"{name} ({url}) not responding")
    if not state.get("api_key", "").strip():
        errors.append("LLM API Key is not set")
    if not user_input:
        errors.append("no computer use request provided")
    return errors
# =============================================================================
# Core processing: send user_input to sampling_loop_sync
# =============================================================================

def process_input(user_input, state):
    # Reset stop flag on new input
    if state["stop"]:
        state["stop"] = False
    if not state.get("skip_validation", False):
        errs = valid_params(user_input, state)
        if errs:
            raise gr.Error("Validation errors: " + ", ".join(errs))
    # Append to history
    state["messages"].append({
        "role": Sender.USER,
        "content": [TextBlock(type="text", text=user_input)]
    })
    state["chatbot_messages"].append((user_input, None))
    yield state["chatbot_messages"]
    # Run loop
    for _ in sampling_loop_sync(
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state["chatbot_messages"], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        max_tokens=16384,
        omniparser_url=args.omniparser_server_url
    ):
        if _ is None or state.get("stop"):
            yield state["chatbot_messages"]
            break
        yield state["chatbot_messages"]

# =============================================================================
# Stop handler
# =============================================================================

def stop_app(state):
    state["stop"] = True
    return "App stopped"

# =============================================================================
# Load header image
# =============================================================================

def get_header_image_base64():
    try:
        img_path = Path(__file__).parent.parent / "imgs" / "header_bar_thin.png"
        data = base64.b64encode(open(img_path,'rb').read()).decode()
        return f'data:image/png;base64,{data}'
    except Exception:
        return None

# =============================================================================
# UI callbacks for settings
# =============================================================================
def update_model(model_selection, state):
    state["model"] = model_selection
    # Determine valid providers
    if model_selection == "claude-3-5-sonnet-20241022":
        choices = [p.value for p in APIProvider if p.value != "openai"]
    elif model_selection in ["omniparser + gpt-4o","omniparser + o1","omniparser + o3-mini"]:
        choices = ["openai"]
    elif model_selection == "omniparser + R1":
        choices = ["groq"]
    elif model_selection == "omniparser + qwen2.5vl":
        choices = ["dashscope"]
    else:
        choices = [p.value for p in APIProvider]
    default = choices[0]
    state["provider"] = default
    # Return UI updates for provider dropdown and API key placeholder
    return (
        gr.update(choices=choices, value=default, interactive=(len(choices)>1)),
        gr.update(placeholder=f"{default.title()} API Key", value=state.get(f"{default}_api_key",""))
    )


def update_only_n_images(val, state):
    state["only_n_most_recent_images"] = val


def update_provider(val, state):
    state["provider"] = val
    return gr.update(value=state.get(f"{val}_api_key",""))


def update_api_key(val, state):
    state["api_key"] = val
    state[f"{state['provider']}_api_key"] = val


def clear_chat(state):
    state["messages"] = []
    state["responses"] = {}
    state["tools"] = {}
    state["chatbot_messages"] = []
    return state["chatbot_messages"]

# =============================================================================
# Build the Gradio UI
# =============================================================================
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("""
        <style>
        .no-padding { padding: 0!important; }
        .markdown-text p { font-size:18px; }
        </style>
    """)
    state = gr.State({})
    setup_state(state.value)

    header_b64 = get_header_image_base64()
    if header_b64:
        gr.HTML(f'<img src="{header_b64}" width="100%">', elem_classes="no-padding")
        gr.HTML('<h1 style="text-align:center;">Omni<span style="font-weight:bold;">Tool</span></h1>')
    else:
        gr.Markdown("# OmniTool")

    if not os.getenv("HIDE_WARNING",False):
        gr.Markdown(INTRO_TEXT, elem_classes="markdown-text")

    # JSON runner controls
    with gr.Row():
        json_dropdown = gr.Dropdown(label="Select JSON File", choices=load_json_files(), interactive=True)
        run_json_button = gr.Button("Run JSON Tasks")

    # Settings accordion
    with gr.Accordion("Settings", open=True):
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=[
                        "omniparser + gpt-4o","omniparser + o1","omniparser + o3-mini",
                        "omniparser + R1","omniparser + qwen2.5vl","claude-3-5-sonnet-20241022"
                    ],
                    value="omniparser + gpt-4o",
                    interactive=True
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0, maximum=10, step=1, value=2, interactive=True
                )
        with gr.Row():
            with gr.Column(scale=1):
                provider = gr.Dropdown(
                    label="API Provider",
                    choices=[p.value for p in APIProvider],
                    value="openai",
                    interactive=False
                )
            with gr.Column(scale=2):
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=state.value.get("api_key",""),
                    placeholder="Paste your API key here",
                    interactive=True
                )

    # Chat input row
    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="Stop", variant="secondary")

    # Chat history and VNC iframe
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=580)
        with gr.Column(scale=3):
            iframe = gr.HTML(
                f'<iframe src="http://{args.windows_host_url}/vnc.html?view_only=1&autoconnect=1&resize=scale" width="100%" height="580" allow="fullscreen"></iframe>',
                elem_classes="no-padding"
            )

    # Bind callbacks
    model.change(fn=update_model, inputs=[model, state], outputs=[provider, api_key])
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    provider.change(fn=update_provider, inputs=[provider, state], outputs=[api_key])
    api_key.change(fn=update_api_key, inputs=[api_key, state], outputs=None)
    chatbot.clear(fn=clear_chat, inputs=[state], outputs=[chatbot])

    submit_button.click(fn=process_input, inputs=[chat_input, state], outputs=[chatbot])
    stop_button.click(fn=stop_app, inputs=[state], outputs=None)
    run_json_button.click(fn=run_json_tasks, inputs=[json_dropdown, state], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7888)
