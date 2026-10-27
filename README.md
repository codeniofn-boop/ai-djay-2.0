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
