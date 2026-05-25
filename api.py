# api.py
# FastAPI server — Hookify AI hook detector
# Supports URL-based detection with ffmpeg conversion

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import httpx
import subprocess
from detect_hook import detect_hook

app = FastAPI(
    title="Hookify AI API",
    description="AI-powered hook detection for songs",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hookify AI API is running! 🐘🎵"}

@app.post("/detect-hook")
async def detect_hook_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = detect_hook(tmp_path)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        os.unlink(tmp_path)

class UrlRequest(BaseModel):
    url: str

@app.post("/detect-hook-url")
async def detect_hook_from_url(body: UrlRequest):
    tmp_original = None
    tmp_wav      = None
    try:
        # Download audio from URL
        async with httpx.AsyncClient() as client:
            response   = await client.get(body.url, timeout=30)
            audio_data = response.content

        # Save original file (m4a/aac)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            tmp.write(audio_data)
            tmp_original = tmp.name

        # Convert to wav using ffmpeg so librosa can read it
        tmp_wav = tmp_original.replace(".m4a", ".wav")
        subprocess.run(
            ["ffmpeg", "-i", tmp_original, "-ar", "22050", "-ac", "1", tmp_wav, "-y"],
            capture_output=True,
            timeout=30
        )

        # Run AI hook detection on wav file
        result = detect_hook(tmp_wav)
        return {"success": True, **result}

    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if tmp_original and os.path.exists(tmp_original):
            os.unlink(tmp_original)
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)