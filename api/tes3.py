import requests
url = "http://localhost:1234/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "deepseek-r1-distill-qwen-7b",  # Use the specified model identifier
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you tailor my resume?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
}

try:
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("Chat Completion Response:", response.json())
    else:
        print(f"Failed to get response. Status Code: {response.status_code}")
        print("Response:", response.text)
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
