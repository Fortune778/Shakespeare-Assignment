import requests
import json

url = "http://localhost:11434/api/generate"

data = {
    "model": "tinyllama",
    "prompt": "Explain microservices architecture",
    "stream": True
}

with requests.post(url, json=data, stream=True) as response:
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode("utf-8"))
            print(chunk.get("response", ""), end="", flush=True)