import time
import json
from typing import Optional, Dict, Any
from groq import Groq  # type: ignore
from config.settings import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


async def groq_inference(
    query: str,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:

    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Satisfy the user request."

    if model_name is None:
        model_name = "openai/gpt-oss-120b"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=8192,
            top_p=1,
            stream=False,
            stop=None,
        )

        response_text = completion.choices[0].message.content

        return response_text

    except Exception as e:
        print(f"Error in Groq inference: {e}")
        return None

