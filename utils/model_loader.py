# # mcp_server/utils/model_loader.py
import mimetypes
import requests
import base64

GEMINI_IMAGE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash"
]
Groq_MODELS = [
    "groq",
    "qwen/qwen3-32b",  # Example Groq model, replace with actual model names as needed
]

OPENROUTER_API_KEY = "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"
GEMINI_API_KEY= "AIzaSyDgZGFgSw6X2L5S7us-OqwJ_AiLq6S1NKg"
GROQ_API_KEY = "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"

def get_openrouter_response(model: str, messages: list) -> str:
    print("DEBUG: Using OpenRouter API key:", OPENROUTER_API_KEY)
    print("Received model:", model)
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages
    }
    res = requests.post(url, json=data, headers=headers)
    if not res.ok:
        raise Exception(f"OpenRouter Error: {res.text}")
    return res.json()["choices"][0]["message"]["content"]
def call_groq_api(messages, model):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages
    }
    response = requests.post(url, json=data, headers=headers)
    print("Groq response status:", response.status_code)
    print("Groq response body:", response.text)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def get_model_response(model, messages, image_urls=None):
    
    if model.lower() in GEMINI_IMAGE_MODELS:
        prompt = messages[-1]["content"]  # last user message
        if image_urls:
            print("Received image_urls:", image_urls)
            return call_gemini_multimodal_api(prompt, image_urls,model)
        else:
            return call_gemini_text_api(prompt, model)
    # Add Groq and other models here as you implement them
    elif model.lower() in Groq_MODELS:
            return call_groq_api(messages,model)
    else:
        # Fallback to your existing OpenRouter or other model logic
        return get_openrouter_response(model, messages)
    
def call_gemini_text_api(prompt, model):
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={GEMINI_API_KEY}"
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }
    response = requests.post(url, json=data)
    print("Gemini response status:", response.status_code)
    print("Gemini response body:", response.text)
    response.raise_for_status()
    result = response.json()
    if "candidates" not in result or not result["candidates"]:
        raise Exception(f"Gemini API returned no candidates: {result}")
    return result["candidates"][0]["content"]["parts"][0]["text"]


def call_gemini_multimodal_api(prompt, image_urls, model):
    """
    Sends a prompt and multiple images to the Gemini multimodal API.
    :param prompt: The user prompt (string)
    :param image_urls: List of image URLs (List[str])
    :param model: The Gemini model name (string)
    :return: The AI's response (string)
    """
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={GEMINI_API_KEY}"
    
    # Build the parts array: first the prompt, then each image as inline_data
    parts = [{"text": prompt}]
    for image_url in image_urls:
        img_response = requests.get(image_url)
        img_data = base64.b64encode(img_response.content).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(image_url)
        if not mime_type:
            mime_type = "image/jpeg"
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": img_data
            }
        })

    data = {
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ]
    }
    print("Sending these images to Gemini:", image_urls)
    print("Sending to Gemini:", data)
    response = requests.post(url, json=data)
    print("Gemini multimodal response status:", response.status_code)
    print("Gemini multimodal response body:", response.text)
    response.raise_for_status()
    result = response.json()
    if "candidates" not in result or not result["candidates"]:
        raise Exception(f"Gemini API returned no candidates: {result}")
    return result["candidates"][0]["content"]["parts"][0]["text"]