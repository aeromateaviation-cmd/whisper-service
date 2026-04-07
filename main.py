from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import time

app = FastAPI()

# Load Whisper model
model = WhisperModel("base", compute_type="int8")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    start_time = time.time()

    print("\n--- New Transcription Request ---", flush=True)

    # Read uploaded file
    content = await file.read()
    print("File name:", file.filename, flush=True)
    print("File size:", len(content), "bytes", flush=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(content)
        tmp.flush()

        # Run Whisper
        segments, info = model.transcribe(tmp.name)

        text = " ".join([seg.text for seg in segments])

    # Log output (THIS shows JSON in Render logs)
    print("Transcript:", text, flush=True)
    print("Transcript length:", len(text), flush=True)
    print("Processing time:", round(time.time() - start_time, 2), "seconds", flush=True)
    print("--- End Request ---\n", flush=True)

    return {"text": text}
