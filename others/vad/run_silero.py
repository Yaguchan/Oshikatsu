import os
import torch
import subprocess
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from speechbrain.pretrained import VAD
from spleeter.separator import Separator


# python others/vad/run_silero.py
NAME = 'kawadame_ebisulive'
VIDEO_PATH = f'data/mp4/{NAME}.mp4'
OUTPUTDIR = f'output/vad/{NAME}' 


def extract_audio(video_path, audio_path="output_audio.wav"):
    """動画から音声をWAV形式で抽出する"""
    command = [
        "ffmpeg", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path


def main():
    
    # output
    os.makedirs(OUTPUTDIR, exist_ok=True)
    
    # audio
    audio_path = extract_audio(VIDEO_PATH)
    
    # separator
    outputdir = f'output/spleeter/{NAME}'
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(audio_path, outputdir)
    
    # Silero VAD
    silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    wav = read_audio(os.path.join(outputdir, 'output_audio/vocals.wav'), sampling_rate=16000)
    timestamps = get_speech_timestamps(wav, silero_model, sampling_rate=16000, return_seconds=True)
    print(timestamps)
    exit()
    
    for i, time_dict in enumerate(timestamps):
        start = time_dict['start']
        duration = time_dict['end'] - time_dict['start']
        output_file = os.path.join(OUTPUTDIR, f'{i}.mp4')
        cmd = [
            "ffmpeg",
            "-y",
            "-i", VIDEO_PATH, 
            "-ss", str(start),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            output_file
        ]
        subprocess.run(cmd)


if __name__ == '__main__':
    main()