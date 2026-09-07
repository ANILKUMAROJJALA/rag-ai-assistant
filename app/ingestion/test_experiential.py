import os

import requests
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("EXPLABS_API_KEY")

if not api_key:
    raise ValueError("EXPLABS_API_KEY was not found.")

response = requests.get(
    "https://api.experientiallabs.ai/v1/models",
    headers={
        "Authorization": f"Bearer {api_key}"
    },
    timeout=30,
)

response.raise_for_status()

data = response.json()

print("Experiential Labs API connected!")
print("\nAvailable models:")

for model in data.get("data", []):
    print("-", model.get("id"))