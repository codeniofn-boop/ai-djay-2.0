#!/usr/bin/env python3
"""
AI DJay 2.0 — single-file remix server + dark neon DJ console.

  python3 app.py            serve on http://0.0.0.0:8000
  python3 app.py --selftest run the multipart and 4/4 quantizer self-tests

Pipeline:  multipart upload  ->  Replicate (MusicGen) remix / evolve  ->  4/4 transient
quantizer (numpy)  ->  WAV served back to the console.  Standard library + numpy only;
ffmpeg is used when present to decode MP3/FLAC/OGG/M4A and non-PCM outputs.

Configuration (environment or a .env file beside this script):
  REPLICATE_API_TOKEN       required for generation (never sent to the browser)
  REPLICATE_MODEL           default meta/musicgen
  REPLICATE_MODEL_VERSION   optional version hash to pin
  REPLICATE_EXTRA_INPUT     optional JSON merged into every prediction input
  PORT                      default 8000
"""
import base64
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
MAX_UPLOAD = 60 * 1024 * 1024
MAX_DATA_URI = 24 * 1024 * 1024
ALLOWED_EXT = {".mp3", ".wav", ".wave", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".aiff", ".aif", ".webm", ".opus"}
API = "https://api.replicate.com/v1"
ID_RE = re.compile(r"^[a-f0-9]{16}$")


def load_dotenv(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(ROOT / ".env")
TOKEN = os.environ.get("REPLICATE_API_TOKEN", "").strip()
MODEL = os.environ.get("REPLICATE_MODEL", "meta/musicgen").strip()
MODEL_VERSION = os.environ.get("REPLICATE_MODEL_VERSION", "").strip()
try:
    EXTRA_INPUT = json.loads(os.environ.get("REPLICATE_EXTRA_INPUT", "") or "{}")
except json.JSONDecodeError:
    EXTRA_INPUT = {}
PORT = int(os.environ.get("PORT", "8000"))
FFMPEG = shutil.which("ffmpeg")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- media registry
MEDIA_LOCK = threading.Lock()
MEDIA: dict = {}  # id -> {"path": Path, "name": str, "kind": "upload"|"output", "ctype": str, "rep_url": str, "rep_at": float}


def new_id() -> str:
    return secrets.token_hex(8)


def register_media(path: Path, name: str, kind: str) -> str:
    mid = new_id()
    ctype = "audio/wav" if path.suffix.lower() in (".wav", ".wave") else (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    with MEDIA_LOCK:
        MEDIA[mid] = {"path": path, "name": name, "kind": kind, "ctype": ctype, "rep_url": "", "rep_at": 0.0, "pcm": None}
    return mid


def get_media(mid: str):
    if not mid or not ID_RE.match(mid):
        return None
    with MEDIA_LOCK:
        return MEDIA.get(mid)


def sniff_audio(head: bytes):
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    return None


# ----------------------------------------------------------------------------- audio io
def ffmpeg_to_pcm(src: Path, dst: Path) -> None:
    if not FFMPEG:
        raise RuntimeError("ffmpeg is not installed and the file is not 16-bit PCM WAV")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dst)],
                   check=True, timeout=600, stdin=subprocess.DEVNULL)


def is_pcm16_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getsampwidth() == 2 and w.getcomptype() == "NONE"
    except Exception:
        return False


def pcm_copy(media: dict) -> Path:
    """Return a 16-bit PCM WAV version of a media item, converting once if needed."""
    if media.get("pcm") and Path(media["pcm"]).exists():
        return Path(media["pcm"])
    src: Path = media["path"]
    if is_pcm16_wav(src):
        media["pcm"] = str(src)
        return src
    dst = src.with_suffix(".pcm.wav")
    ffmpeg_to_pcm(src, dst)
    media["pcm"] = str(dst)
    return dst


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        data = v.astype(np.float32) / 8388608.0
    else:
        raise RuntimeError(f"unsupported WAV sample width {sw}")
    return sr, data.reshape(-1, ch)


def write_wav(path: Path, sr: int, data: np.ndarray) -> None:
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ----------------------------------------------------------------------------- 4/4 quantizer
HOP, WIN = 128, 1024


def band_edges(sr: int, win: int, bands: int = 32):
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    edges = np.geomspace(40.0, min(16000.0, sr / 2), bands + 1)
    idx = np.searchsorted(freqs, edges)
    idx[-1] = len(freqs)
    return [(int(idx[i]), max(int(idx[i]) + 1, int(idx[i + 1]))) for i in range(bands)]


def onset_envelope(mono: np.ndarray, sr: int, hop: int = HOP, win: int = WIN):
    """Band-wise log-power spectral flux (normalised 0..1), frame rate sr/hop.
    Summing power inside 32 log-spaced bands before the log keeps broadband noise
    flat between hits while kicks, snares and hats still jump by tens of dB."""
    if len(mono) < win * 4:
        return None, None, sr / hop
    n = (len(mono) - win) // hop + 1
    window = np.hanning(win).astype(np.float32)
    edges = band_edges(sr, win)
    env = np.zeros(n, dtype=np.float32)
    rms = np.zeros(n, dtype=np.float32)
    prev = None
    chunk = 1024
    for start in range(0, n, chunk):
        count = min(chunk, n - start)
        idx = np.arange(win)[None, :] + hop * (start + np.arange(count))[:, None]
        frames = mono[idx] * window[None, :]
        rms[start:start + count] = np.sqrt((frames ** 2).mean(axis=1))
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        bands = np.stack([power[:, a:b].sum(axis=1) for a, b in edges], axis=1)
        mag = np.log10(bands + 1e-6)
        first = mag[:1] if prev is None else prev
        flux = np.diff(mag, axis=0, prepend=first)
        env[start:start + count] = np.maximum(flux, 0).sum(axis=1)
        prev = mag[-1:]
    env = np.convolve(env, np.ones(3, dtype=np.float32) / 3, mode="same")
    env /= env.max() + 1e-9
    accent = np.maximum(np.diff(rms, prepend=rms[:1]), 0)
    accent = np.convolve(accent, np.ones(3, dtype=np.float32) / 3, mode="same")
    accent /= accent.max() + 1e-9
    return env, accent, sr / hop


def moving_mean(x: np.ndarray, half: int) -> np.ndarray:
    k = 2 * half + 1
    pad = np.pad(x, (half, half), mode="edge")
    c = np.cumsum(np.insert(pad, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def pick_onsets(env: np.ndarray, fr: float, min_gap_s: float = 0.04):
    """Return (times, strengths) of transient peaks with sub-frame precision."""
    n = len(env)
    local = moving_mean(env, max(1, int(fr * 0.35)))
    thr = np.maximum(local * 1.8 + 0.03, 0.1)
    cand = np.where((env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > thr[1:-1]))[0] + 1
    gap = max(1, int(min_gap_s * fr))
    keep = []
    for i in cand:
        if keep and i - keep[-1] < gap:
            if env[i] > env[keep[-1]]:
                keep[-1] = i
            continue
        keep.append(i)
    times, strengths = [], []
    for i in keep:
        a, b, c = env[i - 1], env[i], env[i + 1]
        den = a - 2 * b + c
        d = (a - c) / (2 * den) if den != 0 else 0.0
        times.append((i + float(np.clip(d, -0.5, 0.5))) / fr)
        strengths.append(float(b))
    return np.array(times), np.array(strengths)


def transients(mono: np.ndarray, sr: int):
    """Onset envelope plus transient times referenced to the analysis-window centre."""
    env, accent, fr = onset_envelope(mono, sr)
    if env is None:
        return None, fr, np.array([]), np.array([])
    times, strengths = pick_onsets(env, fr)
    # loudness-aware strength: a kick outranks a quiet hat even when both jump the log flux
    idx = np.clip(np.round(times * fr).astype(int), 0, len(env) - 1)
    loud = np.array([accent[max(0, i - 3): i + 4].max() for i in idx]) if len(idx) else np.array([])
    strengths = strengths * (0.2 + 0.8 * loud)
    tempo_env = env * (0.25 + 0.75 * accent)
    return tempo_env, fr, times + WIN / (2.0 * sr), strengths


def estimate_tempo(env: np.ndarray, fr: float, lo: float = 60.0, hi: float = 200.0):
    x = env - env.mean()
    n = len(x)
    min_lag, max_lag = int(60 * fr / hi), int(60 * fr / lo)
    if n < max_lag * 3:
        return None
    spec = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[: max_lag * 4 + 2]
    ac /= ac[0] + 1e-9
    best, best_lag = -1.0, 0
    for lag in range(min_lag, max_lag + 1):
        bpm = 60 * fr / lag
        score = ac[lag] + 0.5 * ac[2 * lag] + (ac[3 * lag] / 3 if 3 * lag < len(ac) else 0.0) + (0.25 * ac[4 * lag] if 4 * lag < len(ac) else 0.0)
        score += 0.25 * ac[lag // 2] if lag % 2 == 0 and lag // 2 >= min_lag else 0.0
        score *= 0.72 if bpm < 80 else 0.78 if bpm > 170 else 0.9 if (bpm < 95 or bpm > 155) else 1.0
        if score > best:
            best, best_lag = score, lag
    return 60 * fr / best_lag if best_lag > 0 else None


def refine_grid(onsets: np.ndarray, strengths: np.ndarray, beat: float):
    """Lock the beat period and phase onto the strong transients by weighted regression."""
    phase = 0.0
    for _ in range(4):
        bins = 64
        rel = ((onsets - phase) / beat) % 1.0
        hist = np.bincount((rel * bins).astype(int) % bins, weights=strengths, minlength=bins)
        hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
        shift = (np.argmax(hist) + 0.5) / bins
        phase = phase + (shift if shift < 0.5 else shift - 1.0) * beat
        k = np.round((onsets - phase) / beat)
        resid = onsets - (phase + k * beat)
        mask = np.abs(resid) < 0.12 * beat
        if mask.sum() < 3:
            break
        w = np.sqrt(strengths[mask])
        A = np.vstack([k[mask], np.ones(mask.sum())]).T * w[:, None]
        sol = np.linalg.lstsq(A, onsets[mask] * w, rcond=None)[0]
        if 0.25 < sol[0] < 1.2:
            beat, phase = float(sol[0]), float(sol[1])
    return beat, phase


def downbeat_from(onsets: np.ndarray, strengths: np.ndarray, beat: float, phase: float) -> float:
    """Beat 1 = the beat slot (of four) carrying the most transient weight."""
    k = np.round((onsets - phase) / beat)
    near = np.abs(onsets - (phase + k * beat)) < 0.1 * beat
    scores = [strengths[near & ((k % 4) == slot)].sum() for slot in range(4)]
    first = int(k[near].min()) % 4 if near.any() else 0
    best = max(scores)
    # ties (kick on every beat) resolve to the first strong beat so only the pre-roll is trimmed
    slot = next(((first + i) % 4 for i in range(4) if scores[(first + i) % 4] >= 0.85 * best), int(np.argmax(scores)))
    down = phase + slot * beat
    while down - 4 * beat >= 0:
        down -= 4 * beat
    while down < 0:
        down += 4 * beat
    return down


def quantize_to_grid(sr: int, audio: np.ndarray, strength: float = 1.0, bpm_hint: float = None):
    """
    Snap every detected transient onto a strict 4/4 sixteenth-note grid.
    Returns (audio, report).  The first downbeat is moved to 0:00 and the end is padded to a
    whole number of bars so the result loops cleanly.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    mono = audio.mean(axis=1).astype(np.float32)
    env, fr, onsets, strengths = transients(mono, sr)
    report = {"bpm": None, "onsets": 0, "snapped": 0, "mean_shift_ms": 0.0, "max_shift_ms": 0.0, "bars": 0, "note": ""}
    if env is None:
        report["note"] = "too short to analyse"
        return audio, report
    coarse = bpm_hint or estimate_tempo(env, fr)
    if not coarse or len(onsets) < 4:
        report["note"] = "no stable tempo found; left untouched"
        return audio, report
    beat, phase = refine_grid(onsets, strengths, 60.0 / coarse)
    bpm = 60.0 / beat
    step = beat / 4.0
    down = downbeat_from(onsets, strengths, beat, phase)
    sel = onsets >= down - step / 2
    onsets, strengths = onsets[sel], strengths[sel]
    if len(onsets) < 2:
        report.update(bpm=round(bpm, 2), note="fewer than two transients; left untouched")
        return audio, report
    slots = {}
    for o, st in zip(onsets, strengths):
        k = int(round((o - down) / step))
        if k not in slots or st > slots[k][1]:
            slots[k] = (float(o), float(st))
    src_pts, dst_pts, shifts = [0.0], [0.0], []
    for k in sorted(slots):
        o = slots[k][0]
        g = down + k * step
        target = o + (g - o) * strength
        if target <= dst_pts[-1] + 0.004 or o <= src_pts[-1] + 0.004:
            continue
        src_pts.append(o)
        dst_pts.append(target)
        shifts.append(abs(g - o) * strength)
    dur = len(mono) / sr
    tail_shift = dst_pts[-1] - src_pts[-1]
    src_pts.append(dur)
    dst_pts.append(dur + tail_shift)
    src_pts, dst_pts = np.array(src_pts), np.array(dst_pts)
    bar = 4 * beat
    start = down
    end = dst_pts[-1]
    # musical length follows the last transient; a decay tail only adds a bar when it runs
    # more than one beat past that bar line
    last_k = max(slots) if slots else 0
    bars = max(1, last_k // 16 + 1, int(np.ceil((end - start) / bar - 0.25)))
    out_len = int(round(bars * bar * sr))
    t_out = np.arange(out_len) / sr + start
    t_src = np.interp(t_out, dst_pts, src_pts)
    pos = np.clip(t_src * sr, 0, len(mono) - 1)
    base = np.arange(len(mono))
    out = np.zeros((out_len, audio.shape[1]), dtype=np.float32)
    for c in range(audio.shape[1]):
        out[:, c] = np.interp(pos, base, audio[:, c])
    valid = int(min(out_len, max(0, round((dst_pts[-1] - start) * sr))))
    if valid < out_len:
        out[valid:, :] = 0.0
    fade = min(int(0.005 * sr), out_len // 4)
    if fade > 0:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        out[:fade] *= ramp
        if valid > fade:
            out[valid - fade:valid] *= ramp[::-1]
    report.update(
        bpm=round(bpm, 2), onsets=int(len(onsets)), snapped=len(shifts),
        mean_shift_ms=round(float(np.mean(shifts)) * 1000, 1) if shifts else 0.0,
        max_shift_ms=round(float(np.max(shifts)) * 1000, 1) if shifts else 0.0,
        bars=bars, downbeat_trim_ms=round(down * 1000, 1), grid="1/16 at " + str(round(bpm, 2)) + " BPM",
        note="snapped " + str(len(shifts)) + " of " + str(len(onsets)) + " transients at strength " + str(strength))
    return out, report


# ----------------------------------------------------------------------------- replicate client
class ReplicateError(Exception):
    pass


def rep_request(method: str, url: str, data=None, headers=None, timeout: int = 60, raw: bool = False, auth: bool = True):
    h = {"User-Agent": "ai-djay-2.0/1.0"}
    if auth:
        h["Authorization"] = "Bearer " + TOKEN
    if headers:
        h.update(headers)
    body = None
    if data is not None:
        if raw:
            body = data
        else:
            body = json.dumps(data).encode("utf-8")
            h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            payload = r.read()
            return json.loads(payload.decode("utf-8")) if "json" in ctype else payload
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8", "replace"))
            detail = detail.get("detail") or detail.get("error") or json.dumps(detail)[:400]
        except Exception:
            detail = e.reason
        raise ReplicateError(f"Replicate HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise ReplicateError(f"cannot reach {url.split('/')[2]}: {e.reason}") from None
    except TimeoutError:
        raise ReplicateError("request to Replicate timed out") from None


def rep_upload(media: dict, emit) -> str:
    """Hand a local file to Replicate's Files API; cached for 20 hours per media item."""
    if media.get("rep_url") and time.time() - media.get("rep_at", 0) < 20 * 3600:
        return media["rep_url"]
    path: Path = media["path"]
    emit(f"uploading {path.name} ({path.stat().st_size // 1024} KB) to Replicate files…")
    boundary = "----aidjay" + secrets.token_hex(8)
    ctype = media.get("ctype") or "application/octet-stream"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"content\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    try:
        res = rep_request("POST", API + "/files", data=body, raw=True, timeout=600,
                          headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        url = (res.get("urls") or {}).get("get") if isinstance(res, dict) else None
        if not url:
            raise ReplicateError("files API returned no download URL")
    except ReplicateError as e:
        if path.stat().st_size > MAX_DATA_URI:
            raise
        emit(f"files API refused ({e}); falling back to an inline data URI")
        url = f"data:{ctype};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    media["rep_url"], media["rep_at"] = url, time.time()
    return url


def rep_predict(inputs: dict, emit, cancel: threading.Event):
    if MODEL_VERSION:
        res = rep_request("POST", API + "/predictions", {"version": MODEL_VERSION, "input": inputs})
    else:
        res = rep_request("POST", f"{API}/models/{MODEL}/predictions", {"input": inputs})
    pid = res.get("id")
    get_url = (res.get("urls") or {}).get("get") or f"{API}/predictions/{pid}"
    cancel_url = (res.get("urls") or {}).get("cancel")
    emit(f"prediction {pid} created on {MODEL} · status {res.get('status')}")
    t0, seen, delay = time.time(), 0, 1.5
    while True:
        status = res.get("status")
        logs = res.get("logs") or ""
        if len(logs) > seen:
            for line in logs[seen:].splitlines():
                if line.strip():
                    emit("replicate ▸ " + line.strip()[:200])
            seen = len(logs)
        if status in ("succeeded", "failed", "canceled"):
            break
        if cancel.is_set():
            if cancel_url:
                try:
                    rep_request("POST", cancel_url)
                except ReplicateError:
                    pass
            raise ReplicateError("cancelled")
        if time.time() - t0 > 900:
            raise ReplicateError("gave up after 15 minutes of polling")
        time.sleep(delay)
        delay = min(delay * 1.2, 5.0)
        res = rep_request("GET", get_url)
    if status != "succeeded":
        raise ReplicateError(f"prediction {status}: {res.get('error') or 'no detail from Replicate'}")
    out = res.get("output")
    if isinstance(out, list):
        out = out[0] if out else None
    if isinstance(out, dict):
        out = out.get("audio") or out.get("wav") or next(iter(out.values()), None)
    if not isinstance(out, str):
        raise ReplicateError(f"unexpected output shape: {str(out)[:200]}")
    return out, res


def download(url: str, dst: Path, emit) -> None:
    emit("downloading generated audio…")
    if url.startswith("data:"):
        dst.write_bytes(base64.b64decode(url.split(",", 1)[1]))
        return
    payload = rep_request("GET", url, raw=True, timeout=600, auth=False)
    if isinstance(payload, (dict, list)):
        raise ReplicateError("output URL returned JSON instead of audio")
    dst.write_bytes(payload)


# ----------------------------------------------------------------------------- jobs
JOBS_LOCK = threading.Lock()
JOBS: dict = {}


def job_emit(job: dict, line: str) -> None:
    with JOBS_LOCK:
        job["log"].append(f"{datetime.now().strftime('%H:%M:%S')} {line}")
        job["log"] = job["log"][-200:]
    log(f"job {job['id']}: {line}")


def set_stage(job: dict, stage: str, progress: float) -> None:
    with JOBS_LOCK:
        job["stage"], job["progress"] = stage, progress


def run_job(job: dict) -> None:
    emit = lambda line: job_emit(job, line)
    try:
        mode = job["mode"]
        source = get_media(job.get("source_id") or "")
        out_id = new_id()
        if mode == "quantize":
            if not source:
                raise RuntimeError("no source audio to quantize")
            set_stage(job, "decoding", 0.3)
            pcm = pcm_copy(source)
            raw_path = pcm
        else:
            if not TOKEN:
                raise RuntimeError("REPLICATE_API_TOKEN is not set on the server")
            inputs = dict(EXTRA_INPUT)
            inputs.update({"prompt": job["prompt"], "duration": int(job["duration"]),
                           "model_version": job["model_version"], "output_format": "wav",
                           "normalization_strategy": "loudness", "continuation": mode == "evolve"})
            if source:
                set_stage(job, "uploading", 0.05)
                inputs["input_audio"] = rep_upload(source, emit)
                if mode == "evolve":
                    try:
                        src_dur = wav_duration(pcm_copy(source))
                        if src_dur > 30:
                            inputs["continuation_start"] = int(src_dur - 15)
                            inputs["continuation_end"] = int(src_dur)
                            emit(f"continuing from the last 15 s of a {src_dur:.0f} s source")
                    except Exception as e:  # duration is a nicety, not a requirement
                        emit(f"could not measure source length ({e}); continuing from its end")
            elif mode == "evolve":
                raise RuntimeError("evolve needs a remix or an uploaded track to continue from")
            else:
                inputs["model_version"] = "stereo-large" if "melody" in inputs["model_version"] else inputs["model_version"]
                emit("no deck audio: generating from the prompt alone")
            set_stage(job, "generating", 0.15)
            emit("sending prompt: " + job["prompt"][:160])
            out_url, pred = rep_predict(inputs, emit, job["cancel"])
            with JOBS_LOCK:
                job["prediction_id"] = pred.get("id")
                job["metrics"] = pred.get("metrics") or {}
            set_stage(job, "downloading", 0.75)
            raw_path = OUTPUTS / f"{out_id}.raw"
            download(out_url, raw_path, emit)
            set_stage(job, "decoding", 0.82)
            if is_pcm16_wav(raw_path):
                pcm = raw_path.with_suffix(".src.wav")
                raw_path.rename(pcm)
            else:
                pcm = raw_path.with_suffix(".src.wav")
                emit("converting output to 16-bit PCM WAV with ffmpeg")
                ffmpeg_to_pcm(raw_path, pcm)
                raw_path.unlink(missing_ok=True)
        sr, audio = read_wav(pcm)
        report = None
        if job["quantize"]:
            set_stage(job, "quantizing", 0.88)
            emit(f"quantizing {len(audio) / sr:.1f} s onto a strict 4/4 sixteenth grid…")
            audio, report = quantize_to_grid(sr, audio, job["strength"])
            emit(report["note"] + (f" · {report['bpm']} BPM · {report['bars']} bars" if report.get("bpm") else ""))
        final = OUTPUTS / f"{out_id}.wav"
        write_wav(final, sr, audio)
        mid = register_media(final, f"{job['label']}.wav", "output")
        with JOBS_LOCK:
            job["result"] = {
                "id": mid, "file": f"/media/{mid}.wav", "name": f"{job['label']}.wav", "mode": mode, "prompt": job["prompt"],
                "duration_s": round(len(audio) / sr, 2), "sample_rate": sr, "channels": int(audio.shape[1]),
                "source_id": job.get("source_id"), "quantize": report, "created": now_iso(),
                "prediction_id": job.get("prediction_id"), "model": None if mode == "quantize" else MODEL,
                "model_version": None if mode == "quantize" else job["model_version"], "metrics": job.get("metrics") or {},
            }
            job["stage"], job["progress"], job["done"] = "done", 1.0, True
        emit("done → " + job["result"]["file"])
    except Exception as e:
        with JOBS_LOCK:
            job["error"], job["stage"], job["done"] = str(e), "error", True
        emit("error: " + str(e))


def start_job(fields: dict, source_id: str) -> dict:
    mode = fields.get("mode", "blueprint")
    if mode not in ("blueprint", "evolve", "quantize"):
        raise ValueError("unknown mode")
    prompt = (fields.get("prompt") or "").strip()[:2000]
    if mode != "quantize" and not prompt:
        raise ValueError("prompt is empty")
    try:
        duration = int(np.clip(int(float(fields.get("duration", "12"))), 3, 30))
    except ValueError:
        duration = 12
    try:
        strength = float(np.clip(float(fields.get("strength", "1")), 0.0, 1.0))
    except ValueError:
        strength = 1.0
    model_version = fields.get("model_version", "stereo-melody-large")
    if model_version not in ("stereo-melody-large", "stereo-large", "melody-large", "large", "stereo-melody", "melody", "medium", "small"):
        model_version = "stereo-melody-large"
    quantize = fields.get("quantize", "1") not in ("0", "false", "off", "")
    with JOBS_LOCK:
        n = len(JOBS) + 1
    label = {"blueprint": "remix", "evolve": "evolve", "quantize": "quantized"}[mode] + f"_v{n}"
    job = {"id": new_id(), "mode": mode, "prompt": prompt, "duration": duration, "strength": strength, "quantize": quantize,
           "model_version": model_version, "source_id": source_id, "label": label, "stage": "queued", "progress": 0.0,
           "log": [], "done": False, "error": None, "result": None, "created": now_iso(), "cancel": threading.Event()}
    with JOBS_LOCK:
        JOBS[job["id"]] = job
    threading.Thread(target=run_job, args=(job,), daemon=True, name="job-" + job["id"]).start()
    return job


def public_job(job: dict) -> dict:
    with JOBS_LOCK:
        return {k: v for k, v in job.items() if k != "cancel"}


# ----------------------------------------------------------------------------- multipart
def parse_multipart(ctype: str, body: bytes):
    msg = BytesParser(policy=policy.HTTP).parsebytes(b"Content-Type: " + ctype.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ValueError("body is not multipart/form-data")
    fields, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files[name] = (os.path.basename(filename), payload, part.get_content_type())
        else:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, files


def store_upload(filename: str, payload: bytes) -> str:
    if not payload:
        raise ValueError("empty upload")
    if len(payload) > MAX_UPLOAD:
        raise ValueError("file is larger than the 60 MB limit")
    ext = Path(filename).suffix.lower()
    kind = sniff_audio(payload[:16])
    if ext not in ALLOWED_EXT and not kind:
        raise ValueError("not a supported audio file (MP3, WAV, FLAC, OGG, M4A, AIFF)")
    safe_ext = ext if ext in ALLOWED_EXT else "." + (kind or "bin")
    path = UPLOADS / f"{new_id()}{safe_ext}"
    path.write_bytes(payload)
    mid = register_media(path, re.sub(r"[^\w .()\-]+", "_", filename)[:120] or "upload", "upload")
    log(f"stored upload {filename!r} as {path.name} ({len(payload) // 1024} KB, sniffed {kind})")
    return mid


# ----------------------------------------------------------------------------- http
HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI DJay 2.0 · Remix Server</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{color-scheme:dark;--bg:#06080e;--bg2:#0b0f18;--panel:#0f1420;--panel2:#141b2a;--line:#1d2738;--line2:#2b3850;--fg:#e8edf7;--fg2:#9aa7bf;--fg3:#6b778f;
--cyan:#19f0ff;--magenta:#ff2ea6;--lime:#b6ff3b;--amber:#ffb020;--red:#ff3b5c;--glowc:0 0 12px rgba(25,240,255,.55);--glowm:0 0 12px rgba(255,46,166,.55);
--fd:'Chakra Petch','Arial Narrow',sans-serif;--fb:'Barlow','Helvetica Neue',Arial,sans-serif;--fm:'JetBrains Mono',Consolas,monospace}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 var(--fb);-webkit-font-smoothing:antialiased;
background-image:radial-gradient(1200px 520px at 50% -12%,rgba(25,240,255,.08),transparent 60%),radial-gradient(900px 420px at 100% 100%,rgba(255,46,166,.07),transparent 60%)}
.booth{max-width:1480px;margin:0 auto;padding:12px 16px 24px;display:flex;flex-direction:column;gap:12px}
.topbar{display:flex;align-items:center;gap:18px;flex-wrap:wrap;padding:10px 14px;background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:6px;position:relative;overflow:hidden}
.topbar::after{content:"";position:absolute;left:0;right:0;bottom:0;height:2px;background:linear-gradient(90deg,var(--cyan),var(--magenta))}
.brand{display:flex;align-items:center;gap:10px;font-family:var(--fd);font-weight:700;letter-spacing:.04em}
.mark{display:inline-grid;place-items:center;width:36px;height:28px;border:1px solid var(--cyan);color:var(--cyan);border-radius:4px;box-shadow:var(--glowc);font-size:13px}
.name{font-size:22px;text-transform:uppercase}.name em{font-style:normal;color:var(--magenta);text-shadow:var(--glowm)}
.sub{font-family:var(--fm);font-size:10px;color:var(--fg3);letter-spacing:.14em;text-transform:uppercase}
.leds{display:flex;gap:16px;flex-wrap:wrap;margin-left:auto}
.led{display:flex;align-items:center;gap:7px;font-family:var(--fm);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg3)}
.led i{width:9px;height:9px;border-radius:50%;background:#2a3348;flex:none}.led b{color:var(--fg2);font-weight:600}
.led.on i{background:var(--lime);box-shadow:0 0 8px var(--lime)}.led.on b{color:var(--lime)}
.led.warn i{background:var(--amber);box-shadow:0 0 8px var(--amber)}.led.warn b{color:var(--amber)}
.led.bad i{background:var(--red);box-shadow:0 0 8px var(--red)}.led.bad b{color:var(--red)}
.led.busy i{background:var(--magenta);box-shadow:0 0 8px var(--magenta);animation:pulse 1s infinite}.led.busy b{color:var(--magenta)}
@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
.rack{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:12px;align-items:start}
@media (max-width:900px){.rack{grid-template-columns:minmax(0,1fr)}}
.col{display:flex;flex-direction:column;gap:12px;min-width:0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px;display:flex;flex-direction:column;gap:12px;min-width:0;position:relative}
.panel::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;background:repeating-linear-gradient(0deg,rgba(255,255,255,.014) 0 1px,transparent 1px 3px)}
.head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:8px}
.head h2{margin:0;font-family:var(--fd);font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:var(--fg2);display:flex;align-items:center;gap:8px}
.head h2::before{content:"";width:6px;height:6px;background:var(--cyan);box-shadow:var(--glowc);transform:rotate(45deg)}
.m .head h2::before{background:var(--magenta);box-shadow:var(--glowm)}
.chip{font-family:var(--fm);font-size:10px;letter-spacing:.12em;text-transform:uppercase;padding:3px 8px;border:1px solid var(--line2);border-radius:3px;color:var(--fg2);white-space:nowrap}
.chip.ok{border-color:var(--lime);color:var(--lime)}.chip.m{border-color:var(--magenta);color:var(--magenta)}.chip.c{border-color:var(--cyan);color:var(--cyan)}.chip.bad{border-color:var(--red);color:var(--red)}
.drop{display:flex;align-items:center;gap:14px;padding:16px;border:1px dashed var(--line2);border-radius:6px;background:rgba(25,240,255,.03);cursor:pointer;transition:.15s}
.drop:hover,.drop.over{border-color:var(--cyan);box-shadow:var(--glowc)}.drop svg{width:40px;height:40px;color:var(--cyan);flex:none}
.drop b{font-family:var(--fd);font-size:15px;display:block}.drop span{color:var(--fg3);font-size:12px}
.track{font-family:var(--fd);font-size:18px;font-weight:700;overflow-wrap:anywhere}.meta{font-family:var(--fm);font-size:11px;color:var(--fg3)}
canvas.wave{width:100%;height:96px;display:block;border:1px solid var(--line);border-radius:4px;background:var(--bg2)}
audio{width:100%;filter:invert(1) hue-rotate(180deg) saturate(.6);border-radius:4px}
.field{display:flex;flex-direction:column;gap:6px}
.label{font-family:var(--fd);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--fg2)}
textarea{width:100%;resize:vertical;min-height:70px;padding:10px 12px;background:var(--bg2);border:1px solid var(--line2);border-radius:4px;color:var(--fg);font:14px/1.4 var(--fb)}
textarea:focus{outline:none;border-color:var(--magenta);box-shadow:var(--glowm)}textarea::placeholder{color:var(--fg3)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.btn{font-family:var(--fd);font-weight:600;font-size:12px;letter-spacing:.12em;text-transform:uppercase;padding:9px 14px;border-radius:4px;border:1px solid var(--line2);background:var(--panel2);color:var(--fg);cursor:pointer;transition:.15s}
.btn:hover:not(:disabled){border-color:var(--fg3)}.btn:disabled{opacity:.45;cursor:not-allowed}
.btn.c{border-color:var(--cyan);color:var(--cyan);background:rgba(25,240,255,.08)}.btn.c:hover:not(:disabled){box-shadow:var(--glowc)}
.btn.m{border-color:var(--magenta);color:var(--magenta);background:rgba(255,46,166,.08)}.btn.m:hover:not(:disabled){box-shadow:var(--glowm)}
.btn.sm{padding:6px 10px;font-size:11px}
.knob{display:flex;flex-direction:column;gap:4px;min-width:140px;flex:1}
.knob .label{font-size:10px}.knob output{font-family:var(--fm);font-size:16px;color:var(--cyan)}
input[type=range]{width:100%;accent-color:var(--cyan)}select{background:var(--bg2);color:var(--fg);border:1px solid var(--line2);border-radius:4px;padding:7px 8px;font:13px var(--fm)}
.hint{font-family:var(--fm);font-size:11px;color:var(--fg3);flex:1 1 200px}
.console{font-family:var(--fm);font-size:11.5px;background:var(--bg2);border:1px solid var(--line);border-radius:4px;padding:10px;max-height:220px;overflow:auto;color:var(--fg2);white-space:pre-wrap;overflow-wrap:anywhere}
.stage{display:flex;align-items:center;gap:10px;font-family:var(--fm);font-size:12px}
.bar{flex:1;height:6px;background:#1b2434;border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--magenta));transition:width .4s}
.bar.ind i{width:40%;animation:slide 1.2s linear infinite}@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(250%)}}
.err{color:var(--red)}
.versions{display:flex;flex-direction:column;gap:10px}
.ver{border:1px solid var(--line);border-radius:4px;padding:10px;background:var(--bg2);display:flex;flex-direction:column;gap:8px}
.ver.active{border-color:var(--magenta);box-shadow:0 0 10px rgba(255,46,166,.25)}
.ver .t{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.ver .t b{font-family:var(--fd);font-size:14px}
.ver .p{font-size:12px;color:var(--fg2);overflow-wrap:anywhere}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.empty{padding:16px;border:1px dashed var(--line2);border-radius:4px;color:var(--fg3);font-size:13px}
footer{font-family:var(--fm);font-size:11px;color:var(--fg3);display:flex;gap:12px;flex-wrap:wrap;justify-content:space-between}
</style></head><body><div class="booth">
<header class="topbar"><div class="brand"><span class="mark">AI</span><span class="name">DJay <em>2.0</em></span><span class="sub">Remix server</span></div>
<div class="leds"><div class="led" id="ledTok"><i></i><span>Replicate</span><b>…</b></div><div class="led" id="ledModel"><i></i><span>Model</span><b>…</b></div><div class="led" id="ledFf"><i></i><span>ffmpeg</span><b>…</b></div><div class="led" id="ledJob"><i></i><span>Job</span><b>Idle</b></div></div></header>
<main class="rack">
<div class="col">
 <section class="panel"><div class="head"><h2>Deck A · source</h2><span class="chip" id="deckChip">Empty</span></div>
  <div class="drop" id="drop"><input type="file" id="file" accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aiff" hidden>
   <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2"><circle cx="24" cy="24" r="20"/><circle cx="24" cy="24" r="5"/><circle cx="24" cy="24" r="13" stroke-dasharray="3 4"/></svg>
   <div><b id="dropT">Drop MP3 or WAV here</b><span id="dropS">or click to browse · up to 60 MB · stays on this server</span></div></div>
  <div class="track" id="trackName" hidden></div><div class="meta" id="trackMeta"></div>
  <canvas class="wave" id="wave"></canvas><audio id="deckAudio" controls preload="metadata" hidden></audio>
  <div class="row"><button class="btn sm" id="btnQuant" disabled>Quantize deck only (no AI)</button><span class="hint" id="deckHint">Source for Blueprint remixes. Evolve continues whichever version is selected.</span></div>
 </section>
 <section class="panel"><div class="head"><h2>Job console</h2><span class="chip" id="jobChip">Idle</span></div>
  <div class="stage"><span id="stageTxt">No job running</span><div class="bar" id="bar"><i></i></div><span id="elapsed">0 s</span><button class="btn sm" id="btnCancel" hidden>Stop</button></div>
  <div class="console" id="log">Upload a track, write a style blueprint, hit Generate. Replicate logs stream here.</div>
 </section>
</div>
<div class="col">
 <section class="panel m"><div class="head"><h2>AI style blueprint</h2><span class="chip m" id="modelChip">…</span></div>
  <label class="field"><span class="label">AI Style Blueprint Prompt</span><textarea id="pBlue" placeholder="Any genre or artist vibe in the world, e.g. “John Summit tech house with a driving sub-bassline, 126 BPM, F minor” or “Aggressive EDM frat remix with massive supersaw festival chords”"></textarea></label>
  <div class="row">
   <div class="knob"><span class="label">Duration <output id="oDur">12 s</output></span><input type="range" id="dur" min="3" max="30" value="12"></div>
   <div class="knob"><span class="label">Model</span><select id="mver"><option value="stereo-melody-large">stereo-melody-large (follows the deck)</option><option value="melody-large">melody-large</option><option value="stereo-large">stereo-large (text only)</option><option value="large">large (text only)</option></select></div>
   <div class="knob"><span class="label">4/4 quantize <output id="oStr">100%</output></span><input type="range" id="str" min="0" max="100" value="100"></div>
  </div>
  <div class="row"><button class="btn c" id="btnGen">Generate remix</button><span class="hint">MusicGen renders up to 30 s per pass; chain passes with Evolve. Transients are snapped to a sixteenth grid and the file is padded to whole bars.</span></div>
 </section>
 <section class="panel m"><div class="head"><h2>Evolve this remix</h2><span class="chip" id="evoChip">Nothing selected</span></div>
  <label class="field"><span class="label">Evolve This Remix</span><textarea id="pEvo" placeholder="Follow-ups, e.g. “Continue for another 30 seconds with a heavy tech house kick” or “Add a high-pass vocal filter build over 8 bars then drop”"></textarea></label>
  <div class="row"><button class="btn m" id="btnEvo" disabled>Evolve</button><span class="hint">Continues the selected version (or the deck) through MusicGen continuation and quantizes the result as a new version.</span></div>
 </section>
 <section class="panel"><div class="head"><h2>Versions</h2><span class="chip" id="verCount">0</span></div>
  <div class="versions" id="versions"><div class="empty">Remixes land here as numbered versions with BPM, bar count and quantize stats. Select one to evolve it, download it, or put it back on the deck.</div></div>
 </section>
</div>
</main>
<footer><span>Upload → Replicate MusicGen → 4/4 sixteenth-grid quantizer → WAV. Token stays on the server.</span><span id="foot"></span></footer>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=s=>{const m=Math.floor(s/60),x=s-m*60;return m+':'+(x<10?'0':'')+x.toFixed(1)};
const deck={file:null,id:null,name:''}; let versions=[],selected=null,job=null,timer=null,t0=0;
function led(id,cls,txt){const e=$(id);e.className='led '+cls;e.querySelector('b').textContent=txt}
async function health(){try{const h=await(await fetch('/api/health')).json();led('ledTok',h.token?'on':'bad',h.token?'Token set':'No token');led('ledModel','on',h.model);led('ledFf',h.ffmpeg?'on':'warn',h.ffmpeg?'Ready':'Missing');$('modelChip').textContent=h.model;$('foot').textContent='Serving on port '+h.port+' · outputs in '+h.outputs;}catch(e){led('ledTok','bad','Server?')}}
health();
/* deck */
const drop=$('drop'),file=$('file');
drop.addEventListener('click',()=>file.click());
['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',e=>armFile(e.dataTransfer.files[0]));file.addEventListener('change',()=>{armFile(file.files[0]);file.value=''});
async function armFile(f){if(!f)return;deck.file=f;deck.id=null;deck.name=f.name;$('trackName').hidden=false;$('trackName').textContent=f.name;$('trackMeta').textContent=(f.size/1048576).toFixed(2)+' MB · '+(f.type||'audio')+' · not uploaded yet';$('deckChip').textContent='Armed';$('deckChip').className='chip ok';$('dropT').textContent='Drop another file to replace';$('btnQuant').disabled=false;$('btnEvo').disabled=false;$('evoChip').textContent='Evolve from: deck';
 const a=$('deckAudio');a.hidden=false;a.src=URL.createObjectURL(f);drawWave(await decode(f));}
async function decode(blob){try{const buf=await blob.arrayBuffer();const oc=new OfflineAudioContext(1,1,44100);const d=await oc.decodeAudioData(buf);$('trackMeta').textContent+=' · '+fmt(d.duration)+' · '+d.numberOfChannels+' ch';return d}catch(e){return null}}
function drawWave(d,canvas){const c=canvas||$('wave');const dpr=devicePixelRatio||1;c.width=c.clientWidth*dpr;c.height=c.clientHeight*dpr;const g=c.getContext('2d');g.scale(dpr,dpr);const w=c.clientWidth,h=c.clientHeight;g.fillStyle='#0b0f18';g.fillRect(0,0,w,h);if(!d){g.fillStyle='#6b778f';g.font='11px JetBrains Mono';g.textAlign='center';g.fillText('NO WAVEFORM',w/2,h/2+4);return}
 const ch=d.getChannelData(0),n=ch.length;let peaks=new Float32Array(w),mx=0;for(let x=0;x<w;x++){const a=Math.floor(x*n/w),b=Math.floor((x+1)*n/w);let m=0;for(let i=a;i<b;i+=Math.max(1,(b-a)>>9|0)){const v=Math.abs(ch[i]);if(v>m)m=v}peaks[x]=m;if(m>mx)mx=m}
 for(let x=0;x<w;x++){const bh=Math.max(1,peaks[x]/(mx||1)*h*.9);g.fillStyle=x%2?'#19f0ff':'#2a9bb0';g.fillRect(x,h/2-bh/2,1,bh)}}
/* jobs */
async function submit(mode,prompt){if(job&&!job.done)return;const fd=new FormData();fd.append('mode',mode);fd.append('prompt',prompt);fd.append('duration',$('dur').value);fd.append('model_version',$('mver').value);fd.append('strength',($('str').value/100).toString());fd.append('quantize',$('str').value==='0'?'0':'1');
 if(mode==='evolve'&&selected){fd.append('source_id',selected)}else if(deck.id){fd.append('source_id',deck.id)}else if(deck.file){fd.append('file',deck.file,deck.file.name)}
 if(mode==='quantize'&&!deck.file&&!deck.id){logLine('Arm a deck first.');return}
 setBusy(true);logClear();logLine('submitting '+mode+' job…');
 try{const r=await fetch('/api/remix',{method:'POST',body:fd});const j=await r.json();if(!r.ok){throw new Error(j.error||r.statusText)}if(j.upload_id){deck.id=j.upload_id;$('trackMeta').textContent=$('trackMeta').textContent.replace('not uploaded yet','stored as '+j.upload_id)}job={id:j.job_id,done:false};t0=Date.now();poll();}
 catch(e){logLine('error: '+e.message,true);setBusy(false)}}
function setBusy(b){$('btnGen').disabled=b;$('btnEvo').disabled=b||(!selected&&!deck.file&&!deck.id);$('btnQuant').disabled=b||(!deck.file&&!deck.id);$('btnCancel').hidden=!b;$('bar').classList.toggle('ind',b);led('ledJob',b?'busy':'on',b?'Running':'Idle');$('jobChip').textContent=b?'Running':'Idle';$('jobChip').className='chip '+(b?'m':'');if(!b){$('bar').querySelector('i').style.width='0'}}
function logClear(){$('log').textContent=''}function logLine(s,err){const l=$('log');const d=document.createElement('div');if(err)d.className='err';d.textContent=s;l.appendChild(d);l.scrollTop=l.scrollHeight}
async function poll(){if(!job)return;try{const r=await fetch('/api/jobs/'+job.id);const j=await r.json();$('stageTxt').textContent=j.stage;$('elapsed').textContent=Math.round((Date.now()-t0)/1000)+' s';const i=$('bar').querySelector('i');if(j.stage!=='generating'&&j.stage!=='uploading'){$('bar').classList.remove('ind');i.style.width=Math.round(j.progress*100)+'%'}else{$('bar').classList.add('ind')}
 const l=$('log');l.textContent='';(j.log||[]).forEach(s=>logLine(s,/error:/.test(s)));
 if(j.done){job.done=true;setBusy(false);if(j.error){$('stageTxt').textContent='error';$('jobChip').textContent='Error';$('jobChip').className='chip bad';led('ledJob','bad','Error')}else{addVersion(j.result)}return}
 timer=setTimeout(poll,1500)}catch(e){timer=setTimeout(poll,3000)}}
$('btnCancel').addEventListener('click',()=>{if(job)fetch('/api/jobs/'+job.id+'/cancel',{method:'POST'})});
$('btnGen').addEventListener('click',()=>{const p=$('pBlue').value.trim();if(!p){$('pBlue').focus();return}submit('blueprint',p)});
$('btnEvo').addEventListener('click',()=>{const p=$('pEvo').value.trim();if(!p){$('pEvo').focus();return}submit('evolve',p)});
$('btnQuant').addEventListener('click',()=>submit('quantize',''));
$('dur').addEventListener('input',()=>$('oDur').textContent=$('dur').value+' s');$('str').addEventListener('input',()=>$('oStr').textContent=$('str').value+'%');
/* versions */
function addVersion(res){versions.unshift(res);selectVersion(res.id);renderVersions();const a=document.querySelector('#versions audio');if(a){a.play().catch(()=>{})}}
function selectVersion(id){selected=id;$('evoChip').textContent='Evolve from: '+(versions.find(v=>v.id===id)||{}).name;$('evoChip').className='chip m';$('btnEvo').disabled=false;document.querySelectorAll('.ver').forEach(v=>v.classList.toggle('active',v.dataset.id===id))}
function renderVersions(){const box=$('versions');box.textContent='';$('verCount').textContent=versions.length;if(!versions.length){box.innerHTML='<div class="empty">No versions yet.</div>';return}
 versions.forEach((v,i)=>{const n=versions.length-i;const d=document.createElement('div');d.className='ver'+(v.id===selected?' active':'');d.dataset.id=v.id;
  const t=document.createElement('div');t.className='t';const b=document.createElement('b');b.textContent='v'+n+' · '+v.name;t.appendChild(b);const c=document.createElement('span');c.className='chip '+(v.mode==='evolve'?'m':v.mode==='quantize'?'':'c');c.textContent=v.mode;t.appendChild(c);const dur=document.createElement('span');dur.className='chip';dur.textContent=fmt(v.duration_s);t.appendChild(dur);d.appendChild(t);
  if(v.prompt){const p=document.createElement('div');p.className='p';p.textContent='“'+v.prompt+'”';d.appendChild(p)}
  const cv=document.createElement('canvas');cv.className='wave';d.appendChild(cv);
  const au=document.createElement('audio');au.controls=true;au.preload='metadata';au.src=v.file;d.appendChild(au);
  const q=v.quantize;const ch=document.createElement('div');ch.className='chips';const chip=(txt,cls)=>{const s=document.createElement('span');s.className='chip '+(cls||'');s.textContent=txt;ch.appendChild(s)};
  if(q&&q.bpm){chip(q.bpm+' BPM','c');chip(q.bars+' bars','c');chip(q.snapped+'/'+q.onsets+' transients snapped');chip('mean shift '+q.mean_shift_ms+' ms · max '+q.max_shift_ms+' ms');chip('grid '+q.grid)}else if(q){chip('quantize: '+q.note,'')}else{chip('not quantized')}
  if(v.prediction_id)chip('replicate '+v.prediction_id.slice(0,8));d.appendChild(ch);
  const r=document.createElement('div');r.className='row';const mk=(txt,cls,fn)=>{const x=document.createElement('button');x.className='btn sm '+cls;x.textContent=txt;x.addEventListener('click',fn);r.appendChild(x)};
  mk('Evolve from this','m',()=>{selectVersion(v.id);$('pEvo').focus()});mk('Put on deck','',()=>{deck.file=null;deck.id=v.id;deck.name=v.name;$('trackName').hidden=false;$('trackName').textContent=v.name;$('trackMeta').textContent='server media '+v.id+' · '+fmt(v.duration_s);$('deckAudio').hidden=false;$('deckAudio').src=v.file;$('deckChip').textContent='Armed · '+v.name;$('deckChip').className='chip ok';$('btnQuant').disabled=false;fetch(v.file).then(r=>r.blob()).then(decode).then(drawWave)});
  const a=document.createElement('a');a.className='btn sm';a.href=v.file;a.download=v.name;a.textContent='Download WAV';r.appendChild(a);d.appendChild(r);box.appendChild(d);
  fetch(v.file).then(r=>r.blob()).then(b=>decode(b)).then(dd=>drawWave(dd,cv)).catch(()=>{});});}
drawWave(null);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AIDJay/2.0"

    def log_message(self, fmt, *args):  # quieter access log
        log(f"{self.address_string()} {fmt % args}")

    def _json(self, status: int, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str, download_name: str = "") -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        status = 200
        if rng and rng.startswith("bytes="):
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                end = min(end, size - 1)
                status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            self.send_header("Content-Disposition", f'inline; filename="{download_name}"')
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/health":
            return self._json(200, {"ok": True, "token": bool(TOKEN), "model": MODEL, "model_version": MODEL_VERSION or "latest",
                                    "ffmpeg": bool(FFMPEG), "port": PORT, "outputs": str(OUTPUTS), "jobs": len(JOBS)})
        if path == "/api/jobs":
            with JOBS_LOCK:
                items = [{"id": j["id"], "stage": j["stage"], "done": j["done"], "mode": j["mode"], "created": j["created"],
                          "result": j["result"], "error": j["error"]} for j in JOBS.values()]
            return self._json(200, {"jobs": items})
        m = re.match(r"^/api/jobs/([a-f0-9]{16})$", path)
        if m:
            job = JOBS.get(m.group(1))
            return self._json(200, public_job(job)) if job else self._json(404, {"error": "no such job"})
        m = re.match(r"^/media/([a-f0-9]{16})(?:\.[a-z0-9]+)?$", path)
        if m:
            media = get_media(m.group(1))
            if not media or not Path(media["path"]).exists():
                return self._json(404, {"error": "no such media"})
            return self._file(Path(media["path"]), media["ctype"], media["name"])
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        m = re.match(r"^/api/jobs/([a-f0-9]{16})/cancel$", path)
        if m:
            job = JOBS.get(m.group(1))
            if not job:
                return self._json(404, {"error": "no such job"})
            job["cancel"].set()
            return self._json(200, {"ok": True})
        if path != "/api/remix":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return self._json(411, {"error": "missing Content-Length"})
        if length > MAX_UPLOAD + 1024 * 1024:
            return self._json(413, {"error": "request larger than the 60 MB upload limit"})
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return self._json(415, {"error": "send multipart/form-data"})
        body = self.rfile.read(length)
        try:
            fields, files = parse_multipart(ctype, body)
            upload_id = None
            if "file" in files:
                filename, payload, _ = files["file"]
                upload_id = store_upload(filename, payload)
            source_id = upload_id or (fields.get("source_id") or "").strip()
            if source_id and not get_media(source_id):
                return self._json(400, {"error": "unknown source_id"})
            job = start_job(fields, source_id)
            return self._json(202, {"job_id": job["id"], "upload_id": upload_id, "source_id": source_id})
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:
            log(f"request failed: {e!r}")
            return self._json(500, {"error": "server error: " + str(e)})


# ----------------------------------------------------------------------------- selftest
def selftest() -> int:
    ok = True
    # multipart round trip with binary payload that contains CRLFs and high bytes
    blob = bytes(secrets.randbits(8) for _ in range(4000)) + b"\r\n--x\r\n" + bytes(range(256)) * 20
    boundary = "----selftest" + secrets.token_hex(4)
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\nquantize\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"loop.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + blob + f"\r\n--{boundary}--\r\n".encode()
    fields, files = parse_multipart(f"multipart/form-data; boundary={boundary}", body)
    same = files["file"][1] == blob and fields["mode"] == "quantize"
    print(f"multipart round-trip: {'OK' if same else 'FAIL'} ({len(blob)} bytes)")
    ok &= same
    # quantizer: jittered clicks at 126 BPM must land on the grid
    sr, bpm, bars = 44100, 126.0, 8
    beat = 60 / bpm
    rng = np.random.default_rng(7)
    n = int((bars * 4 * beat + 0.4) * sr)
    audio = rng.normal(0, 0.003, (n, 2)).astype(np.float32)
    ideal, jittered = [], []
    pre = 0.137
    for b in range(bars * 4 * 2):  # eighth notes
        t = pre + b * beat / 2
        j = t + rng.uniform(-0.03, 0.03)
        ideal.append(t)
        jittered.append(j)
        i = int(j * sr)
        if i + 4000 < n:
            env = np.exp(-np.arange(4000) / (sr * 0.02)).astype(np.float32)
            env[-800:] *= np.linspace(1, 0, 800, dtype=np.float32)
            tone = np.sin(2 * np.pi * (60 if b % 2 == 0 else 2000) * np.arange(4000) / sr).astype(np.float32)
            audio[i:i + 4000, :] += (env * tone * (0.9 if b % 2 == 0 else 0.4))[:, None]
    out, rep = quantize_to_grid(sr, audio, 1.0)
    _, _, onsets, _ = transients(out.mean(axis=1).astype(np.float32), sr)
    est_beat = 60 / rep["bpm"]
    err_after = np.abs(((onsets / (est_beat / 4)) + 0.5) % 1 - 0.5) * (est_beat / 4) * 1000
    jit = np.array(jittered) - np.array(ideal)
    _, _, on_in, _ = transients(audio.mean(axis=1).astype(np.float32), sr)
    true_hits = sum(1 for t in on_in if np.min(np.abs(np.array(jittered) - t)) < 0.012)
    print(f"quantizer: detected {rep['bpm']} BPM (true {bpm}), {rep['bars']} bars, snapped {rep['snapped']}/{rep['onsets']}; input picker {true_hits} real / {len(on_in) - true_hits} false of {len(jittered)} clicks")
    print(f"  jitter before: mean {np.abs(jit).mean() * 1000:.1f} ms · after: mean {err_after.mean():.1f} ms, max {err_after.max():.1f} ms (grid tolerance 3 ms)")
    good = abs(rep["bpm"] - bpm) < 0.6 and err_after.mean() < 3.0 and rep["bars"] == bars
    print(f"quantizer: {'OK' if good else 'FAIL'}")
    ok &= good
    return 0 if ok else 1


def main() -> None:
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if not TOKEN:
        log("WARNING: REPLICATE_API_TOKEN is not set; generation jobs will fail until it is (quantize-only still works)")
    log(f"model {MODEL}{'@' + MODEL_VERSION if MODEL_VERSION else ''} · ffmpeg {'found' if FFMPEG else 'missing'} · data in {DATA}")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    log(f"AI DJay 2.0 remix server listening on http://0.0.0.0:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")


if __name__ == "__main__":
    main()
