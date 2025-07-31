import mimetypes
import requests
import base64
import os
import json
import asyncio
from typing import AsyncGenerator

from google import genai
from utils.context_utils import extract_text_from_image, extract_text_from_pdf

# Load .env if present (optional, but recommended)
from dotenv import load_dotenv
load_dotenv()

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
    "qwen/qwen3-32b",
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    raise ValueError("GEMINI_API_KEY environment variable is not set or is invalid. Please set it in your environment or .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

def get_openrouter_response(model: str, messages: list) -> str:
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

async def get_openrouter_streaming_response(model: str, messages: list) -> AsyncGenerator[str, None]:
    """Get streaming response from OpenRouter API"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    
    try:
        with requests.post(url, json=data, headers=headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        json_str = line[6:]  # Remove 'data: ' prefix
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(json_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"Error in streaming: {str(e)}"

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
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

async def call_groq_streaming_api(messages, model) -> AsyncGenerator[str, None]:
    """Get streaming response from Groq API"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    
    try:
        with requests.post(url, json=data, headers=headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        json_str = line[6:]  # Remove 'data: ' prefix
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(json_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"Error in streaming: {str(e)}"

import os
from PyPDF2 import PdfReader

def upload_file_to_gemini(file_url):
    response = requests.get(file_url)
    filename = file_url.split("/")[-1]
    with open(filename, "wb") as f:
        f.write(response.content)
    # Check file size
    print("Downloaded file size:", os.path.getsize(filename))
    # Check PDF page count
    try:
        reader = PdfReader(filename)
        print("PDF page count:", len(reader.pages))
    except Exception as e:
        print("PDF integrity check failed:", e)
    # Upload to Gemini
    file_obj = client.files.upload(file=filename, config={'display_name': filename})
    os.remove(filename)
    print("Uploaded file to Gemini:", file_obj)
    return file_obj

async def get_gemini_streaming_response(model: str, messages: list, image_urls=None) -> AsyncGenerator[str, None]:
    """Get streaming response from Gemini"""
    try:
        prompt = messages[-1]["content"]
        file_objs = []
        
        if image_urls:
            for url in image_urls:
                if url.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
                    try:
                        file_objs.append(upload_file_to_gemini(url))
                    except Exception as e:
                        print(f"Error uploading file {url} to Gemini: {e}")
        
        contents = file_objs + [prompt]
        
        # Gemini streaming - using generate_content with stream=True
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={'stream': True}
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"Error in Gemini streaming: {str(e)}"

def get_model_response(model, messages, image_urls=None):
    model_lower = model.lower()
    if model_lower in ["gemini-2.5-pro", "gemini-2.5-flash"]:
        prompt = messages[-1]["content"]
        file_objs = []
        if image_urls:
            for url in image_urls:
                if url.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
                    try:
                        file_objs.append(upload_file_to_gemini(url))
                    except Exception as e:
                        print(f"Error uploading file {url} to Gemini: {e}")
        contents = file_objs + [prompt]
        response = client.models.generate_content(
            model=model,
            contents=contents
        )
        return response.text
    elif model_lower in Groq_MODELS:
        return call_groq_api(messages, model)
    else:
        # Fallback: extract text from PDFs/images and prepend to prompt
        context = ""
        if image_urls:
            for url in image_urls:
                if url.lower().endswith(".pdf"):
                    context += extract_text_from_pdf(url)
                elif url.lower().endswith((".jpg", ".jpeg", ".png")):
                    context += extract_text_from_image(url)
        prompt = messages[-1]["content"]
        full_prompt = f"Context:\n{context}\n\nUser: {prompt}" if context else prompt
        messages[-1]["content"] = full_prompt
        return get_openrouter_response(model, messages)

async def get_streaming_response(model, messages, image_urls=None) -> AsyncGenerator[str, None]:
    """Get streaming response based on model type"""
    model_lower = model.lower()
    
    if model_lower in ["gemini-2.5-pro", "gemini-2.5-flash"]:
        async for chunk in get_gemini_streaming_response(model, messages, image_urls):
            yield chunk
    elif model_lower in Groq_MODELS:
        async for chunk in call_groq_streaming_api(messages, model):
            yield chunk
    else:
        # For other models, use OpenRouter streaming
        context = ""
        if image_urls:
            for url in image_urls:
                if url.lower().endswith(".pdf"):
                    context += extract_text_from_pdf(url)
                elif url.lower().endswith((".jpg", ".jpeg", ".png")):
                    context += extract_text_from_image(url)
        
        prompt = messages[-1]["content"]
        full_prompt = f"Context:\n{context}\n\nUser: {prompt}" if context else prompt
        messages[-1]["content"] = full_prompt
        
        async for chunk in get_openrouter_streaming_response(model, messages):
            yield chunk

# import mimetypes
# import requests
# import base64
# import os

# from google import genai
# from utils.context_utils import extract_text_from_image, extract_text_from_pdf

# # Load .env if present (optional, but recommended)
# from dotenv import load_dotenv
# load_dotenv()

# GEMINI_IMAGE_MODELS = [
#     "gemini-2.5-pro",
#     "gemini-2.5-flash",
#     "gemini-2.5-flash-lite-preview-06-17",
#     "gemini-2.0-flash",
#     "gemini-2.0-flash-lite",
#     "gemini-1.5-flash"
# ]
# Groq_MODELS = [
#     "groq",
#     "qwen/qwen3-32b",
# ]

# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
#     raise ValueError("GEMINI_API_KEY environment variable is not set or is invalid. Please set it in your environment or .env file.")

# client = genai.Client(api_key=GEMINI_API_KEY)

# def get_openrouter_response(model: str, messages: list) -> str:
#     url = "https://openrouter.ai/api/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": model,
#         "messages": messages
#     }
#     res = requests.post(url, json=data, headers=headers)
#     if not res.ok:
#         raise Exception(f"OpenRouter Error: {res.text}")
#     return res.json()["choices"][0]["message"]["content"]

# def call_groq_api(messages, model):
#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": model,
#         "messages": messages
#     }
#     response = requests.post(url, json=data, headers=headers)
#     response.raise_for_status()
#     return response.json()["choices"][0]["message"]["content"]

# import os
# from PyPDF2 import PdfReader

# def upload_file_to_gemini(file_url):
#     response = requests.get(file_url)
#     filename = file_url.split("/")[-1]
#     with open(filename, "wb") as f:
#         f.write(response.content)
#     # Check file size
#     print("Downloaded file size:", os.path.getsize(filename))
#     # Check PDF page count
#     try:
#         reader = PdfReader(filename)
#         print("PDF page count:", len(reader.pages))
#     except Exception as e:
#         print("PDF integrity check failed:", e)
#     # Upload to Gemini
#     file_obj = client.files.upload(file=filename, config={'display_name': filename})
#     os.remove(filename)
#     print("Uploaded file to Gemini:", file_obj)
#     return file_obj

# def get_model_response(model, messages, image_urls=None):
    
#     model_lower = model.lower()
#     if model_lower in ["gemini-2.5-pro", "gemini-2.5-flash"]:
#         prompt = messages[-1]["content"]
#         file_objs = []
#         if image_urls:
#             for url in image_urls:
#                 if url.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
#                     try:
#                         file_objs.append(upload_file_to_gemini(url))
#                     except Exception as e:
#                         print(f"Error uploading file {url} to Gemini: {e}")
#         contents = file_objs + [prompt]
#         response = client.models.generate_content(
#             model=model,
#             contents=contents
#         )
#         return response.text
#     elif model_lower in Groq_MODELS:
#         return call_groq_api(messages, model)
#     else:
#         # Fallback: extract text from PDFs/images and prepend to prompt
#         context = ""
#         if image_urls:
#             for url in image_urls:
#                 if url.lower().endswith(".pdf"):
#                     context += extract_text_from_pdf(url)
#                 elif url.lower().endswith((".jpg", ".jpeg", ".png")):
#                     context += extract_text_from_image(url)
#         prompt = messages[-1]["content"]
#         full_prompt = f"Context:\n{context}\n\nUser: {prompt}" if context else prompt
#         messages[-1]["content"] = full_prompt
#         return get_openrouter_response(model, messages)