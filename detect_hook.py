# detect_hook.py
# AI Hook Detector — the core AIML project of Hookify
# This script analyzes an audio file and finds the hook (catchiest part)
# 
# How it works:
# 1. Load the audio file using librosa
# 2. Extract audio features (energy, melody, rhythm)
# 3. Find the most repeated and energetic section
# 4. Return the start and end timestamp of the hook
#
# This replaces our manual hookStart/hookEnd timestamps!

import librosa          # Audio analysis library
import numpy as np      # Numerical computing
import json             # For outputting results as JSON

def detect_hook(audio_path: str, hook_duration: float = 15.0) -> dict:
    """
    Detect the hook (catchiest part) of a song.
    
    Args:
        audio_path: Path to the audio file (.mp3, .wav, .m4a etc)
        hook_duration: How long the hook should be in seconds (default 15s)
    
    Returns:
        Dictionary with hook_start, hook_end, and confidence score
    """
    
    print(f"🎵 Loading audio: {audio_path}")
    
    # ── Step 1: Load the audio file ───────────────────────────────
    # librosa loads audio as a numpy array
    # y = audio samples, sr = sample rate (usually 22050 Hz)
    y, sr = librosa.load(audio_path, duration=60)  # Load first 60 seconds
    
    print(f"✅ Audio loaded! Duration: {len(y)/sr:.1f}s, Sample rate: {sr}Hz")
    
    # ── Step 2: Extract RMS Energy ────────────────────────────────
    # RMS = Root Mean Square = measure of loudness/energy at each moment
    # High energy usually means chorus/hook
    rms = librosa.feature.rms(y=y)[0]
    
    # ── Step 3: Extract Spectral Centroid ─────────────────────────
    # Spectral centroid = "brightness" of sound
    # Higher centroid = more treble = typically more melodic/exciting
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # ── Step 4: Extract Onset Strength ───────────────────────────
    # Onset = moment when a new note/beat starts
    # High onset strength = lots of rhythmic activity = energetic section
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    
    # ── Step 5: Normalize all features to 0-1 range ──────────────
    # So they can be combined fairly
    def normalize(arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val == min_val:
            return arr
        return (arr - min_val) / (max_val - min_val)
    
    rms_norm             = normalize(rms)
    centroid_norm        = normalize(spectral_centroid)
    onset_norm           = normalize(onset_strength)
    
    # ── Step 6: Align all features to same length ─────────────────
    # Different features have different frame counts — we align them
    min_len = min(len(rms_norm), len(centroid_norm), len(onset_norm))
    rms_norm      = rms_norm[:min_len]
    centroid_norm = centroid_norm[:min_len]
    onset_norm    = onset_norm[:min_len]
    
    # ── Step 7: Combine features into one "hookiness" score ───────
    # We weight energy highest, then melody, then rhythm
    # These weights can be tuned — this is the ML part!
    hookiness = (
        0.5 * rms_norm +          # 50% weight on energy
        0.3 * centroid_norm +     # 30% weight on brightness/melody  
        0.2 * onset_norm          # 20% weight on rhythmic activity
    )
    
    # ── Step 8: Convert frames to time ────────────────────────────
    # librosa works in "frames" — we convert to seconds
    hop_length = 512  # Default librosa hop length
    times = librosa.frames_to_time(
        np.arange(len(hookiness)), 
        sr=sr, 
        hop_length=hop_length
    )
    
    # ── Step 9: Find the best window for the hook ─────────────────
    # We slide a window of hook_duration seconds across the song
    # and find where the average hookiness score is highest
    
    hook_frames = int(hook_duration * sr / hop_length)  # Convert seconds to frames
    
    best_score      = -1
    best_start_frame = 0
    
    # Slide window across the song
    for i in range(len(hookiness) - hook_frames):
        # Average hookiness score for this window
        window_score = np.mean(hookiness[i:i + hook_frames])
        
        if window_score > best_score:
            best_score       = window_score
            best_start_frame = i
    
    # ── Step 10: Convert best frame to timestamp ──────────────────
    hook_start = float(times[best_start_frame])
    hook_end   = hook_start + hook_duration
    
    # Make sure hook doesn't go past the song
    song_duration = len(y) / sr
    if hook_end > song_duration:
        hook_end   = song_duration
        hook_start = max(0, hook_end - hook_duration)
    
    # Round to 1 decimal place
    hook_start = round(hook_start, 1)
    hook_end   = round(hook_end, 1)
    
    print(f"🎯 Hook detected!")
    print(f"   Start: {hook_start}s")
    print(f"   End:   {hook_end}s")
    print(f"   Confidence: {best_score:.3f}")
    
    return {
        "hook_start":  hook_start,
        "hook_end":    hook_end,
        "confidence":  round(float(best_score), 3),
        "song_duration": round(song_duration, 1),
    }


# ── Run directly for testing ──────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 detect_hook.py <audio_file>")
        print("Example: python3 detect_hook.py song.mp3")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    result = detect_hook(audio_file)
    
    print("\n📊 Result (JSON):")
    print(json.dumps(result, indent=2))