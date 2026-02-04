import dashscope
from dashscope import Generation


def call_qwen(
    user_prompt: str,
    model: str = "qwen-plus",
    temperature: float = 0.7
) -> str:
    """
    调用 Qwen LLM
    """
    dashscope.api_key = "your apikey"
    response = Generation.call(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        result_format="message",
    )

    if response.status_code != 200:
        raise RuntimeError(response)

    return response.output.choices[0].message.content
