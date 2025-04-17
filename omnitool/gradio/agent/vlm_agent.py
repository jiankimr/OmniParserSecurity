import json
from collections.abc import Callable
from typing import cast
import uuid
from PIL import Image, ImageDraw
import base64
from io import BytesIO

from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import (
    BetaMessage, BetaTextBlock, BetaToolUseBlock,
    BetaMessageParam, BetaUsage
)

from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
from agent.llm_utils.utils import is_image_path
import time
import re


"""
json-driven 실행 루프에서 각 task 전에 agent.reset_context()
호출 시 완전 초기화(컨텍스트(히스토리, 카운터))된 상태에서 순차 실행 가능
"""
OUTPUT_DIR = "./tmp/outputs"

def extract_data(input_string, data_type):
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    matches = re.findall(pattern, input_string, re.DOTALL)
    return matches[0][0].strip() if matches else input_string

class VLMAgent:
    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str,
        output_callback: Callable,
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
    ):
        # 모델 매핑
        if model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1":
            self.model = "o1"
        elif model == "omniparser + o3-mini":
            self.model = "o3-mini"
        else:
            raise ValueError(f"Model {model} not supported")

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0

        # 대화 히스토리(옵션)
        self.history: list[dict] = []

    def reset_context(self):
        """
        토큰/비용 카운터와 히스토리를 초기화합니다.
        각 JSON task 실행 전 반드시 호출하세요.
        """
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0
        self.history.clear()

    def __call__(self, messages: list, parsed_screen: dict):
        # 스텝 카운트
        self.step_count += 1
        # (옵션) 히스토리에 기록
        self.history.extend(messages)

        # 스크린샷 등 정보 추출
        image_base64 = parsed_screen.get('original_screenshot_base64', '')
        latency_omniparser = parsed_screen.get('latency', 0)
        self.output_callback(f'-- Step {self.step_count}: --', sender="bot")
        screen_info = str(parsed_screen.get('screen_info', []))
        screenshot_uuid = parsed_screen.get('screenshot_uuid', '')
        screen_width = parsed_screen.get('width', 1)
        screen_height = parsed_screen.get('height', 1)

        # 시스템 프롬프트 생성
        system = self._get_system_prompt(parsed_screen.get('screen_info', []))

        # 메시지 전처리: 오래된 이미지 제거
        planner_messages = list(messages)
        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        # 스크린샷 파일 경로 추가
        if isinstance(planner_messages[-1], dict):
            content = planner_messages[-1].setdefault('content', [])
            if isinstance(content, list):
                content.append(f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png")
                content.append(f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png")

        # LLM 호출
        start = time.time()
        if any(k in self.model for k in ["gpt", "o1", "o3-mini"]):
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1",
                temperature=0,
            )
        elif "r1" in self.model:
            vlm_response, token_usage = run_groq_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
        elif "qwen" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=min(2048, self.max_tokens),
                provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0,
            )
        else:
            raise ValueError(f"Model {self.model} not supported")

        # 사용량/비용 업데이트
        self.total_token_usage += token_usage
        rate_map = {'gpt':2.5, 'o1':15, 'o3-mini':1.1, 'r1':0.99, 'qwen':2.2}
        for key, price in rate_map.items():
            if key in self.model:
                self.total_cost += token_usage * price / 1_000_000
                break

        latency_vlm = time.time() - start
        self.output_callback(
            f"LLM: {latency_vlm:.2f}s, OmniParser: {latency_omniparser:.2f}s",
            sender="bot"
        )

        if self.print_usage:
            print(
                f"Total tokens: {self.total_token_usage}, "
                f"Cost: ${self.total_cost:.5f}"
            )

        # JSON 결과 파싱
        # response_str = extract_data(vlm_response, "json")
        # vlm_json = json.loads(response_str)
# JSON 결과 파싱 (실패 시 기본값으로 대체)
        response_str = extract_data(vlm_response, "json").strip()
        try:
            vlm_json = json.loads(response_str)
        except json.JSONDecodeError:
            print(f"[VLMAgent] JSON 디코드 실패, response_str={response_str!r}")
            vlm_json = {
                "Reasoning": "JSON 파싱에 실패했습니다.",
                "Next Action": "None"
            }

        # 박스 시각화
        img_base64 = parsed_screen.get("som_image_base64", "")
        if "Box ID" in vlm_json:
            try:
                bbox = parsed_screen['parsed_content_list'][int(vlm_json['Box ID'])]['bbox']
                cx = int((bbox[0] + bbox[2]) / 2 * screen_width)
                cy = int((bbox[1] + bbox[3]) / 2 * screen_height)
                vlm_json['box_centroid_coordinate'] = [cx, cy]

                raw = base64.b64decode(img_base64)
                img = Image.open(BytesIO(raw))
                draw = ImageDraw.Draw(img)
                r = 10
                draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill='red')
                buf = BytesIO()
                img.save(buf, format="PNG")
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception:
                pass

        # 이미지 & 세부정보 출력
        self.output_callback(
            f'<img src="data:image/png;base64,{img_base64}">',
            sender="bot"
        )
        self.output_callback(
            f'<details><summary>Parsed Elements</summary>'
            f'<pre>{screen_info}</pre></details>',
            sender="bot"
        )

        # Reasoning + Next Action 문자열화
        plan = ""
        for k, v in vlm_json.items():
            if k == "Reasoning":
                plan += v
            else:
                plan += f"\n{k}: {v}"

        # 도구 실행 블록 생성
        blocks = [BetaTextBlock(text=plan, type='text')]
        if 'box_centroid_coordinate' in vlm_json:
            blocks.append(
                BetaToolUseBlock(
                    id=f"toolu_{uuid.uuid4()}",
                    input={'action':'mouse_move','coordinate':vlm_json['box_centroid_coordinate']},
                    name='computer', type='tool_use'
                )
            )
        action = vlm_json.get("Next Action","None")
        if action == "type":
            blocks.append(
                BetaToolUseBlock(
                    id=f"toolu_{uuid.uuid4()}",
                    input={'action':'type','text':vlm_json.get('value','')},
                    name='computer', type='tool_use'
                )
            )
        elif action and action != "None":
            blocks.append(
                BetaToolUseBlock(
                    id=f"toolu_{uuid.uuid4()}",
                    input={'action':action},
                    name='computer', type='tool_use'
                )
            )

        msg = BetaMessage(
            id=f"toolu_{uuid.uuid4()}",
            content=blocks,
            model='', role='assistant', type='message',
            stop_reason='tool_use',
            usage=BetaUsage(input_tokens=0, output_tokens=0)
        )
        return msg, vlm_json

    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)

    def _get_system_prompt(self, screen_info: str = ""):
        main = f"""
You are using a Windows device.
You can only interact via mouse & keyboard on the GUI.
Here are detected boxes: {screen_info}

Available Next Actions:
- type, left_click, right_click, double_click, hover,
  scroll_up, scroll_down, wait, None

Output JSON with:
  Reasoning, Next Action, Box ID (opt), value (opt)
"""
        return main

# helper functions
def _remove_som_images(messages):
    for msg in messages:
        cnt = msg.get("content")
        if isinstance(cnt, list):
            msg['content'] = [
                c for c in cnt
                if not (isinstance(c, str) and 'som' in c and is_image_path(c))
            ]

def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    if images_to_keep is None:
        return messages
    total = 0
    for msg in messages:
        for c in msg.get('content', []):
            if (isinstance(c, str) and is_image_path(c)) or \
               (isinstance(c, dict) and c.get('type')=='tool_result'):
                total += 1
    to_remove = total - images_to_keep
    for msg in messages:
        if to_remove <= 0:
            break
        new_content = []
        for c in msg.get('content', []):
            if to_remove > 0 and isinstance(c, str)\
               and is_image_path(c):
                to_remove -= 1
                continue
            if to_remove > 0 and isinstance(c, dict)\
               and c.get('type')=='tool_result':
                entries = []
                for e in c.get('content', []):
                    if to_remove > 0 and isinstance(e, dict)\
                       and e.get('type')=='image':
                        to_remove -= 1
                        continue
                    entries.append(e)
                c['content'] = entries
            new_content.append(c)
        msg['content'] = new_content
    return messages