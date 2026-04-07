from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import time

app = FastAPI()

model = WhisperModel("base", compute_type="int8")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    start_time = time.time()

    print("\n--- New Transcription Request ---")

    # Read file
    content = await file.read()
    print("File name:", file.filename)
    print("File size:", len(content), "bytes")

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(content)
        tmp.flush()

        segments, info = model.transcribe(tmp.name)

        text = " ".join([seg.text for seg in segments])

    # ✅ ADD THESE LOGS (VERY IMPORTANT)
    print("Transcript:", text)
    print("Transcript length:", len(text))
    print("Processing time:", round(time.time() - start_time, 2), "seconds")
    print("--- End Request ---\n")

    return {"text": text}
