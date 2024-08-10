from moviepy.editor import VideoFileClip
import torchaudio
import torch

def load_video(video):
    try:
        video = VideoFileClip(video)
        print("Video loaded successfully.")
        print(f"Duration: {video.duration} seconds")
        return video
    except Exception as e:
        print(f"Error loading video: {e}")


def convert_video2audio(video, audio_name):
    if video.audio is None:
        print("No audio track found in the video.")
    else:
        print("Audio track found.")
        audio = video.audio
        audio.write_audiofile(audio_name)

def prepare_audio_for_whisper(audio_name):
    audio_file_path = audio_name
    waveform, sample_rate = torchaudio.load(audio_file_path)

    # Se l'audio è a 2 canali, lo converto a mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Whisper richiede un sample rate di 16000 Hz
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    waveform = waveform.squeeze().numpy()
    return waveform