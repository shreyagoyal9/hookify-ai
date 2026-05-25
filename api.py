# api.py
# FastAPI server — Hookify AI hook detector
# Now supports both file upload AND URL-based detection!

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import httpx
from detect_hook import detect_hook

app = FastAPI(
    title="Hookify AI API",
    description="AI-powered hook detection for songs",
    version="2.0.0"
)

# Allow requests from our Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
def root():
    return {"message": "Hookify AI API is running! 🐘🎵"}

# ── Detect hook from file upload ───────────────────────────────────
@app.post("/detect-hook")
async def detect_hook_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
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

# ── Detect hook from URL ───────────────────────────────────────────
# Next.js sends the iTunes preview URL
# Railway downloads it and runs AI detection
class UrlRequest(BaseModel):
    url: str

@app.post("/detect-hook-url")
async def detect_hook_from_url(body: UrlRequest):
    try:
        # Download audio from iTunes URL
        async with httpx.AsyncClient() as client:
            response = await client.get(body.url, timeout=30)
            audio_data = response.content

        # Save to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".m4a"
        ) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            result = detect_hook(tmp_path)
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return {"success": False, "error": str(e)}