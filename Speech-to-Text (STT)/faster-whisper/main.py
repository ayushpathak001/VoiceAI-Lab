
from faster_whisper import WhisperModel
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os

# Load Whisper model
print("Loading Faster-Whisper model...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Model loaded successfully!")

# Audio settings
SAMPLE_RATE = 16000
DURATION = 15  # Recording duration in seconds

# Record audio
print("\nSpeak something into your microphone!")
print(f"Recording for {DURATION} seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

sd.wait()

print("Recording completed!")

# Save audio
audio_file = os.path.join(
    os.path.dirname(__file__),
    "sample.wav"
)

write(audio_file, SAMPLE_RATE, audio)

print("Audio saved successfully!")

# Speech to Text
print("\nConverting speech into text...")

segments, info = model.transcribe(
    audio_file,
    beam_size=5,
    language=None,
    vad_filter=True,
    condition_on_previous_text=False,
    initial_prompt="This is an Indian speaker using Hindi and English mixed together (Hinglish). Technical words include Python, Artificial Intelligence, Voice Assistant, FastAPI and Machine Learning."
)

print("\nDetected Language:", info.language)
print("Language Probability:", info.language_probability)

print("\nYour Transcription:")

for segment in segments:
    print(segment.text, end=" ")

print("\n")
