from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()

model = WhisperModel("base", compute_type="int8")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()

        segments, info = model.transcribe(tmp.name)

        text = " ".join([seg.text for seg in segments])

    return {"text": text}
