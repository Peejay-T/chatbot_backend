from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_chat_title(message: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You generate short chat titles.\n"
                    "Rules:\n"
                    "- Max 5 words\n"
                    "- No punctuation\n"
                    "- Title Case\n"
                    "- Focus on user intent\n"
                    "- No emojis\n"
                )
            },
            {
                "role": "user",
                "content": f"User message:\n{message}"
            }
        ]
    )

    title = response.output_text.strip();
    return title; 