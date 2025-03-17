import os
import torch
import subprocess
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from speechbrain.pretrained import VAD
from spleeter.separator import Separator


# python others/vad/run_speechbrain.py
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


def preprocess_audio(input_path, output_path, target_sr=16000):
    waveform, sr = torchaudio.load(input_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, target_sr)
    print(f"Processed {input_path}: Mono + {target_sr}Hz -> {output_path}")


def get_speech_segments(speech_probs, frame_length=0.01, threshold=0.2):
    speech_probs = np.array(speech_probs.view(-1))
    speech_indices = np.where(speech_probs > threshold)[0]
    if len(speech_indices) == 0: return []
    segments = []
    start = speech_indices[0]
    for i in range(1, len(speech_indices)):
        if speech_indices[i] != speech_indices[i - 1] + 1:
            end = speech_indices[i - 1]
            segments.append([round(start*frame_length,2), round(end*frame_length,2)])
            start = speech_indices[i]
    segments.append((start*frame_length, speech_indices[-1]*frame_length))
    return segments


def merge_segments(ranges, merge_thres=0.5, delete_thres=0.5):
    if not ranges: return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_thres: merged[-1] = [prev_start, end]
        else: merged.append([start, end])
    new_ranges = []
    for start, end in merged:
        if end - start > merge_thres: new_ranges.append([start, end])
    return new_ranges


def main():
    
    # output
    os.makedirs(OUTPUTDIR, exist_ok=True)
    
    # audio
    audio_path = extract_audio(VIDEO_PATH)
    
    # separator
    outputdir = f'output/spleeter/{NAME}'
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(audio_path, outputdir)
    
    # SpeechBrain VAD
    input_audio = os.path.join(outputdir, 'output_audio/vocals.wav')
    processed_audio = os.path.join(outputdir, 'output_audio/vocals_16k.wav')
    preprocess_audio(input_audio, processed_audio)
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")
    speech_probs = vad.get_speech_prob_file(processed_audio)
    timestamps = get_speech_segments(speech_probs)
    print(timestamps)
    timestamps = merge_segments(timestamps)
    print(timestamps)
    
    for i, time_dict in enumerate(timestamps):
        start, end = time_dict
        duration = end - start
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