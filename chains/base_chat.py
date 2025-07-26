# # # mcp_server/chains/base_chat.py

import os
import shutil
import traceback
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel
import cloudinary
import cloudinary.uploader
from utils.model_loader import get_model_response
from utils.firebase_utils import ChatRequest, ChatResponse, create_new_chat, get_chat_threads, store_message
from utils.firebase_utils import get_chat_messages
from google.cloud import firestore
from utils.context_utils import extract_text_from_pdf, extract_text_from_image
from utils.pdf_vector_store import VECTOR_DIR, search_pdf_context, process_and_store_pdfs
import tempfile

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
        db = firestore.Client()
        files_ref = db.collection("files").where("conversation_id", "==", req.chat_id)
        files = [doc.to_dict() for doc in files_ref.stream()]
        pdfs = [f for f in files if f.get("file_type") == "pdf"]
        for msg in history:
            if msg.get("question"):
                messages.append({"role": "user", "content": msg["question"]})
            if msg.get("answer"):
                messages.append({"role": "assistant", "content": msg["answer"]})
        context_text = ""
        if pdfs:
            pdf_urls = [pdf["file_url"] for pdf in pdfs]
            context_text = search_pdf_context(req.chat_id, req.prompt)
        messages.append({"role": "user", "content": req.prompt})
        if context_text.strip():
            print("Final prompt sent to LLM:", messages[-1]["content"])
            prompt_with_context = f"Context from your PDFs:\n{context_text}\n\nUser: {req.prompt}"
            messages[-1]["content"] = prompt_with_context

        
        # Add the latest user prompt
       

        # --- CONTEXT PIPELINE: If no image_urls, fetch all files for the conversation and use as context ---
        image_urls = req.image_urls
        context_text = ""
        if not image_urls and req.chat_id:
            db = firestore.Client()
            files_ref = db.collection("files").where("conversation_id", "==", req.chat_id)
            files = [doc.to_dict() for doc in files_ref.stream()]
            pdfs = [f for f in files if f.get("file_type") == "pdf"]
            images = [f for f in files if f.get("file_type") in ["jpg", "jpeg", "png", "image/jpeg", "image/png"]]
            # Extract text from PDFs
            # for pdf in pdfs:
            #     context_text += extract_text_from_pdf(pdf["file_url"]) + "\n"
            # Extract text from images
            for img in images:
                context_text += extract_text_from_image(img["file_url"]) + "\n"
            if context_text.strip():
                # Prepend context to the prompt
                prompt_with_context = f"Context from your files:\n{context_text}\n\nUser: {req.prompt}"
                messages[-1]["content"] = prompt_with_context
        # Get AI response
        reply = get_model_response(req.model, messages, image_urls=image_urls)
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
@router.post("/delete_files_for_conversation")
def delete_files_for_conversation(conversation_id: str = Query(...)):
    try:
        db = firestore.Client()
        files_ref = db.collection("files").where("conversation_id", "==", conversation_id)
        docs = files_ref.stream()
        for doc in docs:
            data = doc.to_dict()
            public_id = data.get("public_id")
            if public_id:
                cloudinary.uploader.destroy(public_id, invalidate=True)
            doc.reference.delete()
        # Delete vector store
        faiss_path = os.path.join(VECTOR_DIR, f"{conversation_id}")
        if os.path.exists(faiss_path):
            shutil.rmtree(faiss_path)
        return {"status": "success", "message": "All files deleted for conversation"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting files: {e}")

@router.post("/upload_pdf")
async def upload_pdf(
    chat_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    import datetime
    db = firestore.Client()
    tmp_paths = []
    file_names = []
    try:
        # Save all uploaded files to temp locations
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
                tmp_paths.append(tmp_path)
                file_names.append(file.filename)
                # Save file metadata to Firestore
                db.collection("files").add({
                    "file_name": file.filename,
                    "file_type": "pdf",
                    "file_url": tmp_path,  # Replace with cloud URL if you upload to cloud
                    "conversation_id": chat_id,
                    "uploaded_at": datetime.datetime.utcnow()
                })
        # Process all PDFs and update vector store for this chat
        process_and_store_pdfs(tmp_paths, chat_id)
        # Clean up temp files
        for tmp_path in tmp_paths:
            os.remove(tmp_path)
        return {"status": "success", "message": f"{len(files)} PDF(s) uploaded and processed", "files": file_names}
    except Exception as e:
        # Clean up any temp files if error
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return {"status": "error", "message": str(e)}
