# import openai
# import os

# openai.api_key = os.environ.get("OPENAI_API_KEY")

# models = openai.models.list()

# model_names = [model.id for model in models.data]

# for name in model_names:
#     print(name)


import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key="",    
)

response = client.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=[
        {
            "role": "system",
            "content": "You are an AI assistant who knows everything.",
        },
        {
            "role": "user",
            "content": "Hi"
        },
    ],
)

message = response.choices[0].message.content

print(f"Assistant: {message}")