from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


async def start(stream: bool, client: AIClient) -> None:
    """
    Start an interactive chat session with an AI client.

    This function runs a continuous loop that:
    1. Prompts the user for input
    2. Sends the conversation history to the AI
    3. Displays the AI's response
    4. Maintains conversation context

    The loop continues until the user types 'exit'.

    Args:
        stream (bool): If True, use streaming responses (real-time token display).
                      If False, use synchronous responses (complete response at once).
        client (AIClient): The AI client instance to use for generating responses.
    """
    messages = []
    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        # print(messages)
        if stream == True:
            response_message = await client.stream_response(
                messages=messages
            )
        else:
            response_message = client.response(
                messages=messages
            )

        print(f"Assistant: {response_message}")

        messages.append({"role": "assistant", "content": response_message})
