# AI DJay 2.0

A single-page, browser-only music production dashboard styled as a DJ performance rack.
Everything lives in `index.html`; there is no build step and no backend.

- **Deck A** loads any MP3 or WAV by drag-and-drop or file picker, reads container metadata
  (ID3v2/v1 tags, MPEG frame header, WAV `fmt` chunk), decodes the audio, draws the waveform
  and estimates the BPM. A synthesised 126 BPM tech-house demo loop arms itself on open.
- **Monitor** gives transport, a seekable waveform, a log-scale spectrum analyser and an output
  meter with a clip indicator.
- **Analog FX deck** is a live Web Audio chain: media element → high-pass → low-pass → gain →
  analyser → output. Tempo is 0.5×–2.0× via the media element's pitch-preserving time-stretch.
- **AI console** takes a free-text style blueprint prompt and follow-up modification prompts and
  returns a versioned production blueprint (BPM, key, arrangement timeline, sound design, FX rack
  suggestion). Inside a claude.ai artifact it asks Claude on the viewer's account; elsewhere, or
  when Claude is unavailable, an on-device parser produces the blueprint.

Open `index.html` in a browser, or publish it as a Claude Artifact with the `sample` capability.

## Remix server (`app.py`)

`python3 app.py` serves the same neon DJ console with a Python backend on port 8000:

- **Multipart uploads** (MP3, WAV, FLAC, OGG, M4A, AIFF, up to 60 MB) are sniffed by magic
  bytes, renamed to random ids and stored under `data/uploads/`; the client never supplies a path.
- **Replicate** runs `meta/musicgen` (configurable via `REPLICATE_MODEL` / `REPLICATE_MODEL_VERSION`).
  "AI Style Blueprint Prompt" renders a melody-conditioned remix of the deck; "Evolve This Remix"
  continues the selected version. The server uploads audio through Replicate's Files API, creates
  the prediction, polls it with backoff and streams the model logs to the console.
- **4/4 quantizer** (numpy): band-wise spectral-flux transient detection, autocorrelation tempo
  with regression-refined beat period and phase, downbeat selection, then every transient is
  warped onto a strict sixteenth-note grid. Bar one is moved to 0:00 and the file is padded to
  whole bars. `python3 app.py --selftest` verifies the multipart parser and the quantizer.
- The token is read from `REPLICATE_API_TOKEN` (or a git-ignored `.env`) and never sent to the browser.

Requirements: Python 3.10+, numpy, and ffmpeg on the PATH for anything that is not 16-bit PCM WAV.
