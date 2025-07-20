# ===== FastAPI Backend for Flat Firestore Structure (Minimal, Only Required Logic) =====

import os
from fastapi import FastAPI
from chains.base_chat import router
from dotenv import load_dotenv
load_dotenv()
print (os.getenv("OPENROUTER_API_KEY"))

# ------------------ Main FastAPI App ------------------
app = FastAPI()
app.include_router(router)

#set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\abhilahpatra\service-keys\chatbot-53ecb-firebase-adminsdk-fbsvc-f92323769c.json
#uvicorn main:app --host 0.0.0.0 --port 8000
