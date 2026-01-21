import os
from litellm import completion

# os.environ["OPENAI_API_KEY"] = "your-api-key"

# openai call
response = completion(
    model = "gpt-4o",
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)

print(response)