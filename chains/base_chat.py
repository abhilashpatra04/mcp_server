# # # mcp_server/chains/base_chat.py

import traceback
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from utils.model_loader import get_model_response
from utils.firebase_utils import ChatRequest, ChatResponse, create_new_chat, get_chat_threads, store_message
from utils.firebase_utils import get_chat_messages

router = APIRouter()
class ChatRequest(BaseModel):
    uid: str
    prompt: str
    model: str
    chat_id: str = None
    title: str = "Untitled"
    image_urls: Optional[List[str]] = None 

@router.post("/chat", response_model=ChatResponse)
def handle_chat(req: ChatRequest):
    try:
        # Fetch previous messages for this conversation
        history = get_chat_messages(req.uid, req.chat_id) if req.chat_id else []
        # Build messages array for AI API
        messages = []
        for msg in history:
            if msg.get("question"):
                messages.append({"role": "user", "content": msg["question"]})
            if msg.get("answer"):
                messages.append({"role": "assistant", "content": msg["answer"]})
        # Add the latest user prompt
        messages.append({"role": "user", "content": req.prompt})
        # Get AI response
        reply = get_model_response(req.model, messages, image_urls=req.image_urls)
        # Determine chat_id
        chat_id = req.chat_id
        if not chat_id:
            chat_id = create_new_chat(uid=req.uid, title=req.title)
        # Store user + AI message
        store_message(uid=req.uid, chat_id=chat_id, user_msg=req.prompt, ai_msg=reply)
        return ChatResponse(reply=reply, chat_id=chat_id)
    except Exception as e:
        print("Exception in /chat:", e)
        traceback.print_exc()  # <-- Add this for full stack trace in logs
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.get("/get_chats")
def fetch_chats(uid: str = Query(...)):
    try:
        chats = get_chat_threads(uid)
        return {"chats": chats}
    except Exception as e:
        return {"error": str(e)}

@router.get("/get_messages")
def fetch_messages(uid: str = Query(...), chat_id: str = Query(...)):
    try:
        messages = get_chat_messages(uid, chat_id)
        return {"messages": messages}
    except Exception as e:
        return {"error": str(e)}
