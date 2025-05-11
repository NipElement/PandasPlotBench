import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

models = openai.models.list()

model_names = [model.id for model in models.data]

for name in model_names:
    print(name)
