# api.py
# FastAPI server — connects our AI hook detector to the Next.js frontend
# Run with: uvicorn api:app --reload --port 8000
#
# How it works:
# 1. Next.js sends an audio file to this server
# 2. This server runs detect_hook.py on it
# 3. Returns hook_start and hook_end timestamps
# 4. Next.js uses these to play only the hook part!

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from detect_hook import detect_hook

# ── Create FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title="Hookify AI API",
    description="AI-powered hook detection for songs",
    version="1.0.0"
)

# ── Allow Next.js to call this API ────────────────────────────────
# CORS = Cross Origin Resource Sharing
# Without this, browser blocks requests between different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "https://hookify-app-chi.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check endpoint ─────────────────────────────────────────
# Visit http://localhost:8000 to check if server is running
@app.get("/")
def root():
    return {"message": "Hookify AI API is running! 🐘🎵"}

# ── Hook detection endpoint ───────────────────────────────────────
# POST /detect-hook — send an audio file, get hook timestamps back
@app.post("/detect-hook")
async def detect_hook_endpoint(file: UploadFile = File(...)):
    """
    Upload an audio file and get the hook timestamps back.
    
    Returns:
        hook_start: seconds where hook starts
        hook_end: seconds where hook ends  
        confidence: how confident the model is (0-1)
    """
    
    # Save uploaded file to a temp location
    # We can't process it in memory — librosa needs a file path
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1]  # Keep original extension
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run our AI hook detector on the file!
        result = detect_hook(tmp_path)
        return {
            "success":    True,
            "hook_start": result["hook_start"],
            "hook_end":   result["hook_end"],
            "confidence": result["confidence"],
            "filename":   file.filename,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)