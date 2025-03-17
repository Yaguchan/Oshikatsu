import os
import cv2
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip


# python bin/make_short.py --mov kawadame_ebisulive --name HarukaSakuraba
def get_clip_ranges(lst):
    result = []
    start = None
    for i, val in enumerate(lst):
        if val == 1:
            if start is None:
                start = i
        else:
            if start is not None and i - start > 1:
                result.append([start, i - 1])
            start = None
    if start is not None and len(lst) - start > 1:
        result.append([start, len(lst) - 1])
    return result


def merge_ranges(ranges, merge_thres=15, delete_thres=30):
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_thres:
            merged[-1] = [prev_start, end]  # 範囲を結合
        else:
            merged.append([start, end])
    new_ranges = []
    for start, end in merged:
        if end - start > delete_thres: new_ranges.append([start, end])
    return new_ranges


def merge_videos(output_dir, clip_list):
    output_path = os.path.join(output_dir, "merged.mp4")
    mp4_files = [f'clip_{clip_num}_audio.mp4' for clip_num in clip_list]
    list_file = os.path.join(output_dir, 'file_list.txt')
    with open(list_file, 'w') as f:
        for file in mp4_files: f.write(f"file '{file}'\n")
    command = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file,
        "-c", "copy", output_path
    ]
    subprocess.run(command, check=True)
    os.remove(list_file)


def main(args):
    
    # path
    video_path = f'data/mp4/{args.mov}.mp4'
    track_dir = f'output/labels/{args.mov}/seg'
    name_path = f'output/labels/{args.mov}/id_name.txt'
    output_dir = f'output/short/{args.mov}/{args.name}'
    os.makedirs(output_dir, exist_ok=True)

    # 出力ファイルの設定
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # name/track data
    clip_ids = []
    member_flg = [0 for _ in range(num_frame)]
    with open(name_path, 'r', encoding='utf-8') as f:
        member_names = f.read().splitlines()
    member_ids = os.listdir(track_dir)
    member_ids = [int(member_id.replace('.txt', '')) for member_id in member_ids if '.txt' in member_id]
    member_ids.sort()
    for member_id, member_name in zip(member_ids, member_names):
        if member_name != args.name: continue
        print(member_id, member_name)
        # print(member_name, args.name)
        with open(os.path.join(track_dir, f'{member_id}.txt'), 'r', encoding='utf-8') as f:
            track_data = [[int(value) for value in line.split(' ')] for line in f]
            clip_ids.append(member_id)
            for frame_num in range(track_data[0][0]-1, track_data[-1][0]):
                member_flg[frame_num] = 1
    clip_ranges = get_clip_ranges(member_flg)
    # print(clip_ranges)
    clip_ranges = merge_ranges(clip_ranges)
    # print(clip_ranges)

    # 切り抜き操作
    clip_list = []
    for idx, (start_frame, end_frame) in enumerate(clip_ranges):
        clip_list.append(idx)
        print(start_frame, end_frame)
        temp_video_path = os.path.join(output_dir, f'temp_clip_{idx}.mp4')
        temp_audio_path = os.path.join(output_dir, f'temp_clip_{idx}.aac')
        final_clip_path = os.path.join(output_dir, f'clip_{idx}_audio.mp4')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_count in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        start_time = start_frame / fps
        end_time = end_frame / fps
        os.system(f'ffmpeg -i {video_path} -ss {start_time} -to {end_time} -q:a 0 -map a {temp_audio_path}')
        os.system(f'ffmpeg -y -i {temp_video_path} -i {temp_audio_path} -c:v copy -c:a aac -strict experimental {final_clip_path}')
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
    cap.release()
    merge_videos(output_dir, clip_list)
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mov', type=str, help='動画(mp4)のパスを指定してください', required=True)
    parser.add_argument('--name', type=str, help='切り抜く人物の名前を指定してください', required=True)
    args = parser.parse_args()
    main(args)