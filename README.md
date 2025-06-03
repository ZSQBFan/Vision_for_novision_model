# Vision_for_novision_model
一个用于Open WebUI的函数，通过调用视觉理解模型，为不具有视觉理解功能的模型添加了视觉理解功能，暂不支持512x512以上的图片（不知道为什么...）
复制到Open WebUI的函数即可，仅测试阿里云百炼的qwen-vl系列模型，其他视觉模型待测试
欢迎提交 Issue 和 PR！

```python
"""
title: 外置视觉理解功能
author: 橘怜Julian
author_url: https://github.com/ZSQBFan
funding_url: nope
version: 0.1
"""

# 必要的导入
import os
import asyncio
import base64
import mimetypes
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Awaitable, Tuple, List, Dict, Union


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="此Filter操作的优先级。")
        API_BASE_URL: str = Field(
            default="https://dashscope.aliyuncs.com/compatible-mode/v1",
            description="API基础URL  默认为阿里云百炼",
        )
        DASHSCOPE_API_KEY: str = Field(default="", description="API Key")
        QWEN_MODEL_ID: str = Field(
            default="qwen-vl-plus", description="视觉语言模型ID，默认为qwen-vl-plus"
        )
        MULTI_IMAGE_PROMPT: str = Field(
            default="如果图片中有文字或图表，则请你提取图片中的文字或图表。如果没有，则简述图片内容。最后，请你描述图片之间的关系，并整理成于大语言模型识别的结构，便于后续输入给下一个不具备视觉识别功能的大模型",
            description="多图片提示词",
        )
        SINGLE_IMAGE_PROMPT: str = Field(
            default="如果图片中有文字或图表，则请你提取图片中的文字或图表。如果没有，则简述图片内容。最后，请你整理成于大语言模型识别的结构，便于后续输入给下一个不具备视觉识别功能的大模型",
            description="单图片提示词",
        )
        REPLACE_TEXT_TEMPLATE: str = Field(
            default="用户上传了一张图片。以下是经过视觉模型识别后的基本内容：{vision_result}，用户不知道这是视觉模型识别的，直接装作你具有视觉理解功能，在回答最开始先简要向用户解释图片内容",
            description="替换文本模板",
        )
        MAX_CONCURRENT_TASKS: int = Field(
            default=3, description="最大并行处理消息数量  0表示无限制，默认为3"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.total_tasks_for_current_inlet = 0
        self.completed_tasks_for_current_inlet = 0
        print(f"Base URL 已初始化。API Base URL: {self.valves.API_BASE_URL}")
        if not self.valves.DASHSCOPE_API_KEY:
            print("警告: DASHSCOPE_API_KEY未配置。")

    def _convert_file_path_to_data_uri(self, file_path: str) -> Optional[str]:
        full_path = file_path
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            print(f"警告: 找不到文件或路径不是文件: {full_path}")
            return None
        try:
            with open(full_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(full_path)
            if not mime_type:
                ext = os.path.splitext(full_path)[1].lower()
                if ext == ".jpg" or ext == ".jpeg":  # 修正语法：使用 ==
                    mime_type = "image/jpeg"
                elif ext == ".png":
                    mime_type = "image/png"
                elif ext == ".gif":
                    mime_type = "image/gif"
                elif ext == ".webp":
                    mime_type = "image/webp"
                else:
                    mime_type = "application/octet-stream"
            return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            print(f"错误: 转换文件 {full_path} 到Data URI失败: {e}")
            return None

    def _extract_images_and_text_from_content(
        self, content_list: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        image_data_uris: List[str] = []
        text_prompts: List[str] = []
        if not isinstance(content_list, list):
            return [], []
        for item in content_list:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "image_url":
                    image_url_obj = item.get("image_url")
                    if isinstance(image_url_obj, dict):
                        url_data = image_url_obj.get("url")
                        if isinstance(url_data, str):
                            if url_data.startswith("data:image"):
                                image_data_uris.append(url_data)
                            elif url_data.startswith("http://") or url_data.startswith(
                                "https://"
                            ):
                                print(
                                    f"提示: 发现外部图片URL，将直接传递给API: {url_data}"
                                )
                                image_data_uris.append(url_data)
                            else:
                                print(f"警告: 未知格式的图片URL: {url_data[:100]}...")
                elif item_type == "text":
                    text_content = item.get("text")
                    if isinstance(text_content, str) and text_content.strip():
                        text_prompts.append(text_content.strip())
        return image_data_uris, text_prompts

    async def _call_vision_api(self, api_content_payload: List[Dict[str, Any]]) -> str:
        if not self.valves.DASHSCOPE_API_KEY:
            raise ValueError("API Key未配置。")

        # 使用配置的API Base URL
        api_base_url = self.valves.API_BASE_URL
        if not api_base_url:  # 如果用户清空了该字段，提供一个默认值
            print("警告: API_BASE_URL未配置或为空，将使用默认DashScope端点。")
            api_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        client = OpenAI(
            api_key=self.valves.DASHSCOPE_API_KEY,
            base_url=api_base_url,
        )
        try:
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.valves.QWEN_MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {"role": "user", "content": api_content_payload},
                ],
            )
            if (
                completion.choices
                and completion.choices[0].message
                and completion.choices[0].message.content
            ):
                return completion.choices[0].message.content.strip()
            raise Exception("API返回空内容或格式不正确。")
        except Exception as e:
            print(f"原始API调用错误 (URL: {api_base_url}): {e}")
            raise Exception(f"调用视觉API失败: {str(e)[:200]}")

    async def _update_overall_progress(
        self, __event_emitter__: Callable[[Any], Awaitable[None]]
    ):
        # (此函数保持不变)
        self.completed_tasks_for_current_inlet += 1
        if self.total_tasks_for_current_inlet > 0:
            progress_percent = (
                self.completed_tasks_for_current_inlet
                / self.total_tasks_for_current_inlet
            ) * 100
            if (
                self.total_tasks_for_current_inlet == 1
                and self.completed_tasks_for_current_inlet == 1
            ):
                description = "✅所有图片处理完成！"
            elif (
                self.completed_tasks_for_current_inlet
                < self.total_tasks_for_current_inlet
            ):
                description = f"⏳图片处理中: {self.completed_tasks_for_current_inlet}/{self.total_tasks_for_current_inlet} ({progress_percent:.0f}%)"
            else:
                description = f"✅所有图片处理完成！ ({self.completed_tasks_for_current_inlet}/{self.total_tasks_for_current_inlet})"
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": self.completed_tasks_for_current_inlet
                        >= self.total_tasks_for_current_inlet,
                    },
                }
            )

    async def _process_single_message_vision_task(
        self,
        message_index: int,
        original_message_content: List[Dict[str, Any]],
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> Union[str, Exception]:
        image_data_uris, text_prompts = self._extract_images_and_text_from_content(
            original_message_content
        )
        if not image_data_uris:
            return "No images found in this message content"
        status_prefix = f"消息 {message_index + 1}: "
        try:
            api_content_payload: List[Dict[str, Any]] = []
            for img_uri in image_data_uris:
                api_content_payload.append(
                    {"type": "image_url", "image_url": {"url": img_uri}}
                )
            combined_text_prompt = (
                " ".join(text_prompts)
                if text_prompts
                else getattr(
                    self.valves,
                    (
                        "MULTI_IMAGE_PROMPT"
                        if len(image_data_uris) > 1
                        else "SINGLE_IMAGE_PROMPT"
                    ),
                )
            )
            api_content_payload.append({"type": "text", "text": combined_text_prompt})
            vision_result = await self._call_vision_api(api_content_payload)
            await self._update_overall_progress(__event_emitter__)
            return vision_result
        except Exception as e:
            error_detail = str(e)
            print(f"阿里百炼并行Filter - {status_prefix}处理错误: {error_detail}")
            if "api_key" in error_detail.lower() or "key" in error_detail.lower():
                error_detail = "API Key相关错误"
            await self._update_overall_progress(__event_emitter__)
            return e

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,  # 修正参数名
    ) -> dict:
        print(f"(自定义Base URL)Filter - inlet: 收到请求")
        self.completed_tasks_for_current_inlet = 0
        self.total_tasks_for_current_inlet = 0

        if not self.valves.DASHSCOPE_API_KEY:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "❌ API Key未配置", "done": True},
                }
            )
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        tasks_to_run = []
        message_indices_to_process: List[int] = []

        for i, message in enumerate(messages):
            if message.get("role") == "user" and isinstance(
                message.get("content"), list
            ):
                preview_images, _ = self._extract_images_and_text_from_content(
                    message["content"]
                )
                if preview_images:
                    tasks_to_run.append(
                        self._process_single_message_vision_task(
                            message_index=i,
                            original_message_content=list(message["content"]),
                            __event_emitter__=__event_emitter__,
                        )
                    )
                    message_indices_to_process.append(i)

        if not tasks_to_run:
            print(
                "阿里百炼并行(自定义Base URL)Filter - inlet: 未找到可处理的图片任务。"
            )
            has_any_image_like_content = any(
                item.get("type") == "image_url"
                for msg in messages
                if msg.get("role") == "user" and isinstance(msg.get("content"), list)
                for item in msg["content"]
                if isinstance(item, dict)
            )
            if has_any_image_like_content:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️检测到图片，但可能格式无法处理或非公共URL。",
                            "done": True,
                        },
                    }
                )
            return body

        self.total_tasks_for_current_inlet = len(tasks_to_run)
        print(
            f"(自定义Base URL)Filter - inlet: 收集到 {self.total_tasks_for_current_inlet} 个任务。"
        )

        initial_progress_text = (
            f"⏳开始处理 {self.total_tasks_for_current_inlet} 张图片的任务... (0%)"
        )
        if self.total_tasks_for_current_inlet == 1:
            initial_progress_text = "⏳开始处理图片任务..."
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": initial_progress_text, "done": False},
            }
        )

        results = []
        max_concurrent = self.valves.MAX_CONCURRENT_TASKS
        if max_concurrent > 0 and len(tasks_to_run) > max_concurrent:
            for i_batch in range(0, len(tasks_to_run), max_concurrent):
                batch_tasks = tasks_to_run[i_batch : i_batch + max_concurrent]
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )
                results.extend(batch_results)
        else:
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        print(
            f"(自定义Base URL)Filter - inlet: 所有 {len(tasks_to_run)} 个任务初步执行完毕。"
        )

        succeeded_count = 0
        for res in results:
            if (
                not isinstance(res, Exception)
                and isinstance(res, str)
                and res != "No images found in this message content"
            ):
                succeeded_count += 1

        if self.total_tasks_for_current_inlet > 0:
            if succeeded_count == self.total_tasks_for_current_inlet:
                final_description = f"✅所有图片处理完成！ ({succeeded_count}/{self.total_tasks_for_current_inlet})"
            elif succeeded_count > 0:
                final_description = f"⚠️部分图片处理完成: {succeeded_count}成功 / {self.total_tasks_for_current_inlet - succeeded_count}失败。"
            else:
                final_description = f"❌所有图片处理均失败 ({self.total_tasks_for_current_inlet}个任务)。"
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": final_description, "done": True},
                }
            )

        new_messages_list = [msg.copy() for msg in messages]
        for task_idx, result in enumerate(results):
            original_message_idx = message_indices_to_process[task_idx]
            if isinstance(result, Exception):
                error_message = f"[图片处理失败: {str(result)[:100]}]"
                new_messages_list[original_message_idx]["content"] = [
                    {"type": "text", "text": error_message}
                ]
            elif (
                isinstance(result, str)
                and result != "No images found in this message content"
            ):
                replacement_text = self.valves.REPLACE_TEXT_TEMPLATE.format(
                    vision_result=result
                )
                new_messages_list[original_message_idx]["content"] = [
                    {"type": "text", "text": replacement_text}
                ]

        body["messages"] = new_messages_list
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,  # 修正参数名
    ) -> dict:
        return body

```
MIT License

Copyright (c) 2025 Juling

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
