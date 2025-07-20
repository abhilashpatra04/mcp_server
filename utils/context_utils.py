import requests
import pdfplumber
import tempfile
from utils.model_loader import get_model_response
import pytesseract
from PIL import Image
import io

def extract_text_from_pdf(url: str) -> str:
    # Download the PDF to a temp file
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp.flush()
        text = ""
        with pdfplumber.open(tmp.name) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text


def extract_text_from_image(url: str) -> str:
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    text = pytesseract.image_to_string(img)
    return text