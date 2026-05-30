import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomAnthropicAIClient(AIClient):
    """
    Custom HTTP client for Anthropic's Claude API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Claude's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """
    def __init__(self, endpoint: str, model_name: str, api_key: str, system_prompt: str):
        """
        Initialize the Anthropic client with SDK.

        Args:
            endpoint (str): The Anthropic API endpoint (for compatibility, not used by SDK).
            model_name (str): The Claude model to use (e.g., 'claude-3-opus', 'claude-sonnet-4-5').
            api_key (str): The Anthropic API key for authentication.
            system_prompt (str): The system instruction to guide Claude's behavior.
        """
        #TODO:
        # Call to __init__ of super class
        # Add Anthropic and AsyncAnthropic clients https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#usage
        # (In readme you can find samples with both of these clients)
        # Useful links with request/response samples:
        #   - https://docs.anthropic.com/en/api/overview
        #   - https://docs.anthropic.com/en/api/messages
        super().__init__(
            endpoint="https://open.bigmodel.cn/api/paas/v4/chat/completions",
            model_name=model_name,
            api_key=api_key,
            system_prompt=system_prompt
        )

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no content blocks.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Requires 'x-api-key' header and 'anthropic-version' header.
            Claude's API returns content as an array of content blocks.
            The response is printed to stdout before being returned.
        """
        #TODO:
        # https://docs.anthropic.com/en/api/messages-examples
        # - Prepare headers with api key, anthropic version and content type
        # - Add System prompt
        # - Execute post request to AI API (use `requests`)
        # - Parse response
        # - Print response to console
        # - Return ASSISTANT message
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        system = self._system_prompt
        data = {
            "model": self._model_name,
            "system": system,
            "messages": messages,
            "temperature": 1.0
        }
        response = requests.post(self._endpoint, headers=headers, json=data)

        if response.status_code == 200:
            print(response.json())
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"API call failed: {response.status_code}, {response.text}")

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Anthropic's SSE format, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Listens for 'content_block_delta' events with 'text_delta' type.
            Stops processing when 'message_stop' event is received.
            Each delta is printed to stdout as it arrives.
        """
        #TODO:
        # https://docs.anthropic.com/en/docs/build-with-claude/streaming
        # - Prepare headers with api key, anthropic version and content type
        # - Add System prompt
        # - Execute post request to AI API (use `aihttp`)
        # - Handle stream with chunks
        # - Parse response
        # - Print chunks to console
        # - Return ASSISTANT message
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        system = self._system_prompt
        data = {
            "model": self._model_name,
            "system": system,
            "messages": messages,
            "temperature": 1.0,
            "stream": True
        }
        response_message = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=headers, json=data) as resp:
                async for line in resp.content:
                    if line:
                        text = line.decode().strip()
                        # print(text)
                        # Zhipu streaming responses are usually prefixed with "data: "
                        if text.startswith("data: "):
                            text = text[6:]
                        if text and text != "[DONE]":
                            try:
                                event = json.loads(text)
                                # The content is in event['choices'][0]['delta']['content']
                                delta = event.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content')
                                if content:
                                    print(content, end='', flush=True)
                                    response_message += content
                            except Exception as e:
                                print("Error parsing event:", e, text)
                print()  # Newline after streaming is done
        return response_message
