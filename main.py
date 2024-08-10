import torch
from transformers import pipeline
import numpy as np
from utils import load_video, convert_video2audio, prepare_audio_for_whisper

video_name = "video.mov"
audio_name = "audio.mp3"

video = load_video(video_name)

convert_video2audio(video, audio_name)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    device=device,
)

waveform = prepare_audio_for_whisper(audio_name)


#Transcription start, is divided by chunks due to the audio length
chunk_duration_s = 30
chunk_size = int(16000 * chunk_duration_s)
transcription = []

for i in range(0, len(waveform), chunk_size):
    chunk = waveform[i:i + chunk_size]

    if len(chunk) < chunk_size:
        # Handle the last chunk with padding because its length could be less than chunk_duration_s
        padding = chunk_size - len(chunk)
        chunk = np.pad(chunk, (0, padding), mode='constant')

    prediction = pipe({"array": chunk, "sampling_rate": 16000})["text"]
    transcription.append(prediction)

full_transcription = " ".join(transcription)

print("\nTRANSCRIPTION")
print(full_transcription)
