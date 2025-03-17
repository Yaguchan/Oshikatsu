import os
import torch
import whisper
import subprocess
import torchaudio
import torchaudio.transforms as T
from spleeter.separator import Separator


# python others/run_whisper.py
NAME = 'kawadame'
VIDEO_PATH = f'data/mp4/{NAME}.mp4'
OUTPUTDIR = f'output/whisper/{NAME}' 


def extract_audio(video_path, audio_path="output_audio.wav"):
    """動画から音声をWAV形式で抽出する"""
    command = [
        "ffmpeg", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path


def preprocess_audio(input_path, output_path, target_sr=16000):
    waveform, sr = torchaudio.load(input_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, target_sr)
    print(f"Processed {input_path}: Mono + {target_sr}Hz -> {output_path}")
    

def remove_close_elements(lst, threshold=1.0):
    result = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - result[-1] > threshold:
            result.append(lst[i])
    return result


def main():
    
    # output
    os.makedirs(OUTPUTDIR, exist_ok=True)
    
    # whisper
    model = whisper.load_model("medium") # large-v3
    audio_path = extract_audio(VIDEO_PATH)
    
    # separator
    outputdir = f'output/spleeter/{NAME}'
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(audio_path, outputdir)
    input_audio = os.path.join(outputdir, 'output_audio/vocals.wav')
    processed_audio = os.path.join(outputdir, 'output_audio/vocals_16k.wav')
    preprocess_audio(input_audio, processed_audio)

    result = model.transcribe(processed_audio)
    list_time = []
    list_text = []
    print(result)
    exit()
    for segment in result["segments"]:
        list_time.append(segment["start"])
        # list_end.append(segment["end"])
        list_text.append(segment["text"])
    os.remove(audio_path)
    print(result["segments"])
    print(list_time)
    list_time = remove_close_elements(list_time)
    print(list_time)
    exit()
    
    # split
    for i in range(len(list_time) - 1):
        start = list_time[i]
        duration = list_time[i+1] - start
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