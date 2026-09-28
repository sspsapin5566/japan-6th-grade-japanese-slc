import os
from pathlib import Path
from faster_whisper import WhisperModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

audio_path = Path("output/audio.mp3")
srt_path = Path("output/subtitles.srt")
txt_path = Path("output/transcript.txt")

print("Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

print("Transcribing audio...")
segments, info = model.transcribe(str(audio_path), language="ja", beam_size=5)

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

srt_lines = []
txt_lines = []

for i, seg in enumerate(segments, 1):
    start_str = format_time(seg.start)
    end_str = format_time(seg.end)
    text = seg.text.strip()
    srt_lines.append(f"{i}\n{start_str} --> {end_str}\n{text}\n")
    txt_lines.append(text)
    if i % 100 == 0:
        print(f"Processed {i} segments...")

srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
print("Done writing subtitles and transcript!")
