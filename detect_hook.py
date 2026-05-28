# detect_hook.py
# AI Hook Detector — the core AIML project of Hookify
#
# Algorithm v2 — now with melody repetition detection!
#
# How it works:
# 1. Load the audio file using librosa
# 2. Extract 4 audio features:
#      a. RMS Energy        — hooks are loud/energetic
#      b. Spectral Centroid — hooks are bright/melodic
#      c. Onset Strength    — hooks have strong rhythmic activity
#      d. Chroma Repetition — hooks REPEAT (verse→chorus→verse→chorus)
#         We compute the chroma (melody fingerprint) of each candidate window
#         and measure how often that melodic pattern appears elsewhere in the song.
#         Sections that repeat more are more likely to be the hook.
# 3. Combine into a "hookiness" score with tuned weights
# 4. Slide a 15-second window to find the highest-scoring section
# 5. Return hook_start, hook_end, confidence

import librosa
import numpy as np
import json


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for rows of X (n x d)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8          # avoid div-by-zero
    X_normed = X / norms
    return X_normed @ X_normed.T      # (n x n)


def detect_hook(audio_path: str, hook_duration: float = 15.0) -> dict:
    """
    Detect the hook (catchiest part) of a song.

    Args:
        audio_path:    Path to the audio file (.mp3, .wav, .m4a etc.)
        hook_duration: Length of the hook in seconds (default 15 s)

    Returns:
        dict with hook_start, hook_end, confidence, song_duration
    """

    print(f"🎵 Loading audio: {audio_path}")

    # ── Step 1: Load audio ────────────────────────────────────────
    y, sr = librosa.load(audio_path, duration=60)
    print(f"✅ Loaded  {len(y)/sr:.1f}s  @  {sr} Hz")

    hop_length = 512   # ~23 ms per frame at 22 050 Hz

    # ── Step 2a: RMS Energy ───────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # ── Step 2b: Spectral Centroid ("brightness") ─────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    # ── Step 2c: Onset Strength (rhythmic activity) ───────────────
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # ── Step 2d: Chroma CQT (12-bin melody fingerprint) ──────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)  # (12, n_frames)

    # ── Step 3: Align all features to the same length ────────────
    min_len = min(len(rms), len(centroid), len(onset), chroma.shape[1])
    rms      = rms[:min_len]
    centroid = centroid[:min_len]
    onset    = onset[:min_len]
    chroma   = chroma[:, :min_len]

    # ── Step 4: Normalize frame-level features to [0, 1] ─────────
    def normalize(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return arr if hi == lo else (arr - lo) / (hi - lo)

    rms_n    = normalize(rms)
    cent_n   = normalize(centroid)
    onset_n  = normalize(onset)

    # ── Step 5: Build window-level features ───────────────────────
    hook_frames = int(hook_duration * sr / hop_length)
    n_windows   = min_len - hook_frames
    if n_windows <= 0:
        # Song too short — return full clip
        return {
            "hook_start": 0.0,
            "hook_end":   round(len(y) / sr, 1),
            "confidence": 0.0,
            "song_duration": round(len(y) / sr, 1),
        }

    # Per-window mean of frame features
    rms_w    = np.array([rms_n[i:i + hook_frames].mean()    for i in range(n_windows)])
    cent_w   = np.array([cent_n[i:i + hook_frames].mean()   for i in range(n_windows)])
    onset_w  = np.array([onset_n[i:i + hook_frames].mean()  for i in range(n_windows)])

    # Per-window mean chroma vector (12-dim melody fingerprint)
    chroma_w = np.array([chroma[:, i:i + hook_frames].mean(axis=1) for i in range(n_windows)])  # (n_windows, 12)

    # ── Step 6: Melody repetition score via chroma self-similarity ─
    # sim_matrix[i, j] = cosine similarity between window i and window j
    sim_matrix = _cosine_similarity_matrix(chroma_w)

    # For each window, average similarity to *non-adjacent* windows
    # (adjacent windows overlap heavily — we skip ±1 window away)
    skip = max(1, hook_frames // 2)
    rep_scores = np.zeros(n_windows)
    for i in range(n_windows):
        mask = np.ones(n_windows, dtype=bool)
        lo   = max(0, i - skip)
        hi   = min(n_windows, i + skip + 1)
        mask[lo:hi] = False
        if mask.sum() > 0:
            rep_scores[i] = sim_matrix[i, mask].mean()

    rep_n = normalize(rep_scores)

    # ── Step 7: Combine into hookiness ────────────────────────────
    # Weights: energy 35%, brightness 20%, rhythm 15%, repetition 30%
    hookiness = (
        0.35 * normalize(rms_w)   +   # energy
        0.20 * normalize(cent_w)  +   # brightness
        0.15 * normalize(onset_w) +   # rhythm
        0.30 * rep_n                  # melody repetition ← new!
    )

    # ── Step 8: Find the best window ─────────────────────────────
    best_idx   = int(np.argmax(hookiness))
    best_score = float(hookiness[best_idx])

    # ── Step 9: Convert frame index → timestamp ───────────────────
    times      = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=hop_length)
    hook_start = float(times[best_idx])
    hook_end   = hook_start + hook_duration

    song_duration = len(y) / sr
    if hook_end > song_duration:
        hook_end   = song_duration
        hook_start = max(0.0, hook_end - hook_duration)

    hook_start = round(hook_start, 1)
    hook_end   = round(hook_end,   1)

    print(f"🎯 Hook detected!")
    print(f"   Start:      {hook_start}s")
    print(f"   End:        {hook_end}s")
    print(f"   Confidence: {best_score:.3f}")

    return {
        "hook_start":    hook_start,
        "hook_end":      hook_end,
        "confidence":    round(best_score, 3),
        "song_duration": round(song_duration, 1),
    }


# ── Run directly for testing ──────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 detect_hook.py <audio_file>")
        print("Example: python3 detect_hook.py song.mp3")
        sys.exit(1)

    result = detect_hook(sys.argv[1])
    print("\n📊 Result (JSON):")
    print(json.dumps(result, indent=2))
