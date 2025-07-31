# # mcp_server/tools/firestore_tool.py
# from mcp.server.fastmcp import FastMCP
# from utils.firebase_utils import create_new_chat, store_message,get_chat_threads,get_chat_messages
# from google.cloud.firestore_v1.base_query import FieldFilter

# mcp = FastMCP("FirestoreChatTools")

# @mcp.tool()
# async def create_chat(uid: str, title: str) -> str:
#     """
#     Create a new chat thread for a user.
#     """
#     try:
#         chat_id = create_new_chat(uid, title)
#         return f"Chat created with ID: {chat_id}"
#     except Exception as e:
#         return f"Error creating chat: {e}"

# @mcp.tool()
# async def store_chat_message(uid: str, chat_id: str, user_msg: str, ai_msg: str) -> str:
#     """
#     Store a user and AI message into an existing chat thread.
#     """
#     try:
#         store_message(uid, chat_id, user_msg, ai_msg)
#         return "Message stored successfully"
#     except Exception as e:
#         return f"Error storing message: {e}"

# # @mcp.tool()
# # async def delete_chat_thread(uid: str, chat_id: str) -> str:
# #     """
# #     Delete an entire chat thread and its messages.
# #     """
# #     try:
# #         delete_chat(uid, chat_id)
# #         return "Chat deleted successfully"
# #     except Exception as e:
# #         return f"Error deleting chat: {e}"
# # @mcp.tool()
# # def get_chats(uid: str):
# #     """ Fetch the chats from the database(firestore)"""
# #     db = get_firestore_client()
# #     chats_ref = db.collection("users").document(uid).collection("threads")
# #     docs = chats_ref.order_by("created_at", direction="DESCENDING").stream()
# #     return [
# #         {**doc.to_dict(), "chat_id": doc.id}
# #         for doc in docs
# #     ]
# # @mcp.tool()
# # def get_chat_messages(uid: str, chat_id: str):
# #     """Fetch all messages in a chat with the give chat ID from database(firestore)"""
# #     db = get_firestore_client()
# #     messages_ref = db.collection("users").document(uid).collection("threads").document(chat_id).collection("messages")
# #     docs = messages_ref.order_by("timestamp").stream()
# #     return [
# #         doc.to_dict()
# #         for doc in docs
# #     ]
# # @mcp.tool()
# # def get_chat_messages_tool(uid: str, chat_id: str):
# #     db = get_firestore_client()
# #     messages_ref = db.collection("users").document(uid).collection("chats").document(chat_id).collection("messages")
# #     messages_docs = messages_ref.order_by("timestamp").stream()

# #     messages = []
# #     for doc in messages_docs:
# #         data = doc.to_dict()
# #         messages.append({
# #             "user": data.get("user"),
# #             "ai": data.get("ai"),
# #             "timestamp": data.get("timestamp").isoformat() if data.get("timestamp") else None
# #         })

# #     return {"messages": messages}

# if __name__ == "__main__":
#     mcp.run(transport="streamable-http")



from google.cloud import firestore
from datetime import datetime
import mcp
import cloudinary
import cloudinary.uploader


from utils.context_utils import extract_text_from_image, extract_text_from_pdf
from utils.model_loader import get_model_response

db = firestore.Client()

@mcp.tool()
async def add_file_metadata(uid: str, conversation_id: str, file_url: str, file_type: str, file_name: str, public_id: str) -> str:
    """
    Add file metadata to Firestore for a conversation.
    """
    try:
        doc_ref = db.collection("files").document()
        doc_ref.set({
            "uid": uid,
            "conversation_id": conversation_id,
            "file_url": file_url,
            "file_type": file_type,
            "file_name": file_name,
            "public_id": public_id,
            "uploaded_at": datetime.utcnow()
        })
        return "File metadata added successfully"
    except Exception as e:
        return f"Error adding file metadata: {e}"

@mcp.tool()
async def get_files_for_conversation(conversation_id: str) -> list:
    """
    Fetch all file metadata for a conversation.
    """
    try:
        files_ref = db.collection("files").where("conversation_id", "==", conversation_id)
        docs = files_ref.stream()
        files = [doc.to_dict() for doc in docs]
        return files
    except Exception as e:
        return []

cloudinary.config(
  cloud_name = "dkkyiygll",
  api_key = "713672969564931",
  api_secret = "4A9T1zfrrI5rad0eidhr6DOTsTk"
)

@mcp.tool()
async def delete_file(public_id: str, conversation_id: str) -> str:
    """
    Delete a file from Cloudinary and its metadata from Firestore.
    """
    try:
        # Delete from Cloudinary
        cloudinary.uploader.destroy(public_id, invalidate=True)
        # Delete from Firestore
        files_ref = db.collection("files").where("public_id", "==", public_id).where("conversation_id", "==", conversation_id)
        docs = files_ref.stream()
        for doc in docs:
            doc.reference.delete()
        return "File deleted successfully"
    except Exception as e:
        return f"Error deleting file: {e}"

@mcp.tool()
async def delete_files_for_conversation(conversation_id: str) -> str:
    """
    Delete all files for a conversation from Cloudinary and Firestore.
    """
    try:
        files_ref = db.collection("files").where("conversation_id", "==", conversation_id)
        docs = files_ref.stream()
        for doc in docs:
            data = doc.to_dict()
            public_id = data.get("public_id")
            if public_id:
                cloudinary.uploader.destroy(public_id, invalidate=True)
            doc.reference.delete()
        return "All files deleted for conversation"
    except Exception as e:
        return f"Error deleting files: {e}"
    
@mcp.tool()
async def chat_with_context(conversation_id: str, prompt: str, model: str) -> str:
    """
    Fetch all files for the conversation, extract context, and call the AI model.
    """
    try:
        files = await get_files_for_conversation(conversation_id)
        context = ""
        for file in files:
            if file['file_type'] == 'pdf':
                context += extract_text_from_pdf(file['file_url'])
            elif file['file_type'] in ['jpg', 'png']:
                context += extract_text_from_image(file['file_url'])
        full_prompt = f"Context:\n{context}\n\nUser: {prompt}"
        ai_response = get_model_response(model, [{"role": "user", "content": full_prompt}])        
        return ai_response
    except Exception as e:
        return f"Error in chat_with_context: {e}"
    
