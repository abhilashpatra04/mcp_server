# AI Chat Backend with Virtual Expert Agents

A FastAPI-based backend that provides AI chat functionality with virtual expert agents, web search capabilities, file processing (PDF/images), and Firebase integration.

## Features

- **Virtual Expert Agents**: SQL, AI/ML, Android, Web Development, DevOps, and Blockchain experts
- **Multi-Model Support**: OpenRouter, Gemini, and Groq API integration
- **File Processing**: PDF text extraction and image OCR
- **Web Search**: DuckDuckGo search with content extraction
- **Real-time Chat**: Streaming and non-streaming responses
- **Firebase Integration**: Firestore for chat storage and file metadata
- **Vector Search**: PDF content vectorization with FAISS

## Project Structure

```
├── main.py                     # FastAPI application entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project metadata
├── .env                       # Environment variables (create this)
├── firebase_credentials.json  # Firebase service account (add this)
├── chains/
│   └── base_chat.py           # Main chat router with virtual agents
├── tools/
│   ├── firestore_tool.py      # Firebase/Firestore operations
│   └── web_search_tool.py     # Web search functionality
├── utils/
│   ├── context_utils.py       # PDF/image text extraction
│   ├── firebase_utils.py      # Firebase helper functions
│   ├── model_loader.py        # AI model integrations
│   ├── pdf_vector_store.py    # Vector search for PDFs
│   └── web_scraper.py         # Web content scraping
└── vectorstores/              # Vector database storage
```

## Prerequisites

- Python 3.12+
- Firebase project with Firestore enabled
- API keys for AI services (see below)

## Required Credentials & Setup

### 1. Environment Variables (.env file)

Create a `.env` file in the root directory with the following variables:

```env
# OpenRouter API Key (for various AI models)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Groq API Key
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Firebase Service Account

Create a `firebase_credentials.json` file in the root directory with your Firebase service account credentials.

#### How to get Firebase credentials:

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project (or create a new one)
3. Go to Project Settings → Service Accounts
4. Click "Generate new private key"
5. Download the JSON file and rename it to `firebase_credentials.json`
6. Place it in your project root directory

#### Firebase setup requirements:
- Enable Firestore Database
- Create the following collections (they'll be created automatically on first use):
  - `conversations` - stores chat threads
  - `messages` - stores individual messages
  - `files` - stores file metadata

### 3. API Keys Setup Guide

#### OpenRouter API Key:
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up/login to your account
3. Go to API Keys section
4. Generate a new API key
5. Add it to your `.env` file

#### Google Gemini API Key:
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key"
4. Create a new API key
5. Add it to your `.env` file

#### Groq API Key:
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up/login to your account
3. Navigate to API Keys
4. Create a new API key
5. Add it to your `.env` file

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up credentials** (see sections above)
   - Create `.env` file with API keys
   - Add `firebase_credentials.json` file

6. **Set Firebase credentials path** (Windows example)
   ```bash
   set GOOGLE_APPLICATION_CREDENTIALS=firebase_credentials.json
   ```

## Running the Application

1. **Start the server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Access the API**
   - API will be available at: `http://localhost:8000`
   - Interactive docs: `http://localhost:8000/docs`

## API Endpoints

### POST /chat
Main chat endpoint supporting both streaming and non-streaming responses.

**Request Body:**
```json
{
  "uid": "user_id",
  "prompt": "Your question here",
  "model": "gemini-2.5-pro",
  "chat_id": "optional_existing_chat_id",
  "title": "Chat Title",
  "image_urls": ["optional_image_urls"],
  "web_search": false,
  "agent_type": "SQL_EXPERT",
  "stream": false
}
```

**Available Models:**
- `gemini-2.5-pro`, `gemini-2.5-flash` (Google Gemini)
- `groq`, `qwen/qwen3-32b` (Groq)
- Any OpenRouter supported model

**Available Agent Types:**
- `SQL_EXPERT` - Database and SQL expertise
- `AI_ML_EXPERT` - Machine Learning and AI
- `ANDROID_EXPERT` - Android development
- `WEB_EXPERT` - Full-stack web development
- `DEVOPS_EXPERT` - DevOps and infrastructure
- `BLOCKCHAIN_EXPERT` - Blockchain and DeFi

## Additional Features

### File Upload Support
- PDF files: Automatic text extraction and vector search
- Images: OCR text extraction
- Files stored in Cloudinary with metadata in Firestore

### Web Search Integration
- Real-time web search using DuckDuckGo
- Content extraction from search results
- Automatic source citation

### Vector Search
- PDF content is automatically vectorized using FAISS
- Semantic search across uploaded documents
- Context-aware responses based on document content

## Development

### Adding New Virtual Agents
Edit `chains/base_chat.py` and add new agent definitions to the `VIRTUAL_EXPERT_AGENTS` dictionary.

### Environment Variables
The application uses `python-dotenv` to load environment variables from the `.env` file automatically.

### Database Schema
- **conversations**: `{id, title, createdAt, uid}`
- **messages**: `{id, conversationId, question, answer, createdAt, uid}`
- **files**: `{uid, conversation_id, file_url, file_type, file_name, public_id, uploaded_at}`

## Troubleshooting

1. **Firebase Connection Issues**
   - Ensure `firebase_credentials.json` is in the root directory
   - Verify Firestore is enabled in your Firebase project
   - Check that the service account has proper permissions

2. **API Key Issues**
   - Verify all API keys are correctly set in `.env`
   - Ensure API keys have sufficient credits/quota
   - Check for any typos in environment variable names

3. **Model Loading Issues**
   - Ensure you have the correct model names
   - Check API key permissions for specific models
   - Verify network connectivity to API endpoints

4. **File Processing Issues**
   - Install Tesseract OCR for image text extraction
   - Ensure sufficient disk space for temporary files
   - Check file format compatibility

## Security Notes

- Never commit `.env` or `firebase_credentials.json` to version control
- Use environment variables for all sensitive configuration
- Implement proper authentication and authorization for production use
- Consider rate limiting and input validation for production deployment


