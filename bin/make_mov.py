import os
import cv2
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip


# python bin/make_mov.py --mov 61thSingleDance_64_70_1k --member-enjp-list face_recognition/member_list/member.csv --font data/font/NotoSansJP-Black.ttf


def choose_colors(num_colors):
  tmp = list(plt.colors.CSS4_COLORS.values())
  random.shuffle(tmp)
  label2color = tmp[:num_colors]
  return label2color


def hex_to_bgr(hex_color):
    """16進数カラーコードをBGRに変換"""
    hex_color = hex_color.lstrip("#")  # "#"を削除
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def main(args):
    
    # path
    video_path = f'data/mp4/{args.mov}.mp4'
    track_dir = f'output/labels/{args.mov}/seg'
    name_path = f'output/labels/{args.mov}/id_name.txt'
    if args.member_name == 'all':
        output_path = f'output/annotated_mov/{args.mov}_seg.mp4'
    else:
        output_path = f'output/annotated_mov/{args.mov}_{args.member_name}.mp4'
    os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok=True)

    # 出力ファイルの設定
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # class_jp
    df = pd.read_csv(args.member_enjp_list)
    if args.jp:
        en2jp = df.set_index('en')['jp'].to_dict()
        font = ImageFont.truetype(args.font, 20)
    
    # name data
    with open(name_path, 'r', encoding='utf-8') as f:
        members = f.read().splitlines()
    num_classes = len(set(members))
    # colors = choose_colors(num_classes)
    colors = np.random.randint(0, 255, size=(num_classes, 3))
    
    # track data
    ts = [[] for _ in range(num_frame+1)]
    member_ids = os.listdir(track_dir)
    member_ids = [int(member_id.replace('.txt', '')) for member_id in member_ids if '.txt' in member_id]
    for member_id in member_ids:
        with open(os.path.join(track_dir, f'{member_id}.txt'), 'r', encoding='utf-8') as f:
            track_data = [[int(value) for value in line.split(' ')] for line in f]
            for data in track_data:
                t = data[0]
                seg_points = data[1:]
                seg_mask = [(seg_points[i], seg_points[i+1]) for i in range(0, len(seg_points), 2)]
                seg_mask.append(seg_mask[0])
                ts[t].append([member_id, seg_mask])
            
    # dict
    member_ids = sorted(member_ids)
    id_to_member = {member_id:member for member_id, member in zip(member_ids, members)}
    # member_to_color = {member:color for member, color in zip(set(members), colors)}
    member_to_color = df.set_index('en')['color'].apply(hex_to_bgr).to_dict()
    member_to_color['nan'] = (128, 128, 128)

    # draw bbox
    print('ids + names -> mov')
    frame_count = 0
    pbar = tqdm(total=num_frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.jp: 
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame)
        for member_id, seg_mask in ts[frame_count]:
            member = id_to_member[member_id]
            # unknown member or not choice member
            if member == 'nan' or (args.member_name != 'all' and args.member_name != member): continue
            B, G, R = map(int, member_to_color[member])
            x1, y1, w, h = cv2.boundingRect(np.array(seg_mask))
            x2, y2 = x1+w, y1+h
            if args.jp:
                member = en2jp[member]
                bbox = draw.textbbox((x1, y1), member, font=font)
                # bbox = draw.textbbox((x1, y1), f'{member_id}{member}', font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # draw.line(seg_mask, fill=(R,G,B), width=3)
                # draw.rectangle([x1, y1, x1+w, y1+h], outline=(R,G,B), width=4)
                # draw.rectangle([x1 - 1, y1 - text_height - 10, x1 + text_width + 5, y1], fill=(R,G,B))
                # draw.text((x1 + 5, y1 - text_height - 8), member, font=font)
                draw.rectangle([x1 - 1, y1, x1 + text_width + 5, y1 + text_height + 10], fill=(R,G,B))
                draw.text((x1 + 5, y1 + text_height - 20), member, font=font)
            else:
                cv2.polylines(frame, [np.array(seg_mask)], isClosed=True, color=(B, G, R), thickness=10)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                # cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(member) * 12, y1), (B, G, R), -1)
                cv2.rectangle(frame, (x1 - 1, y1 + 30), (x1 + len(member) * 16, y1), (B, G, R), -1)
                cv2.putText(frame, f'{member}', (x1 + 5, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if args.jp: 
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame)
        frame_count += 1
        pbar.update(1)
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    
    # add audio
    output2_path = output_path.replace('.mp4', '_audio.mp4')
    clip = VideoFileClip(output_path)
    audio = AudioFileClip(video_path)
    audio = audio.set_duration(clip.duration)
    video = clip.set_audio(audio)
    video.write_videofile(output2_path, codec='libx264', audio_codec='aac')
    if os.path.exists(output_path): os.remove(output_path)
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mov', type=str, help='動画(mp4)のパスを指定してください', required=True)
    parser.add_argument('--member-enjp-list', type=str, help='メンバーリスト(.csv)のパスを指定してください', required=True)
    parser.add_argument('--font', type=str, help='フォント(.ttf)のパスを指定してください', required=True)
    parser.add_argument('--jp', default=False, help='名前を日本語表記する場合True/しない場合False')
    parser.add_argument('--seg', default=True, help='名前を日本語表記する場合True/しない場合False')
    parser.add_argument('--member-name', type=str, default='all', help='特定のメンバーのみプロットする場合指定してください')
    args = parser.parse_args()
    main(args)