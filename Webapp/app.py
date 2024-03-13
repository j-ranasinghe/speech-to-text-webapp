from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import tempfile
import wave
from flask import after_this_request


app = Flask(__name__)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


model_id = "distil-whisper/distil-small.en"

# Load the model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Create the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/record', methods=['POST'])
def record():
    duration = 10  # Recording duration 
    sr = 16000  
    channels = 1  

    # Record audio 
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=channels, dtype='int16')
    sd.wait()  
    
    # Save recorded audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        save_wav(audio, tmp_filename, sr)

        # ASR on the recorded audio
        result = pipe(tmp_filename)

    # Clean temporary audio file 
    @after_this_request
    def cleanup(response):
        try:
            os.remove(tmp_filename)
        except Exception as e:
            print("Error cleaning up temporary file:", e)
        return response

    return result["text"]

def save_wav(audio, filename, sr):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())

if __name__ == '__main__':
    app.run(debug=True)
