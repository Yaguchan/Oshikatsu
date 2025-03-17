import os
import cv2
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict
from moviepy.editor import VideoFileClip, AudioFileClip
from ultralytics.utils.plotting import Annotator, colors


# python bin/mov2ids_scene.py --mov emiru --yolo-seg-weights weights/yolo11x-seg.pt --tracking-yaml data/yaml/bytetrack.yaml
def calc_frame_diff(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    diff_mean = np.mean(diff)
    return diff_mean


def main(args):
    
    # yolo model
    model = YOLO(args.yolo_seg_weights)
    
    # text/video setting
    text_outdir = f'./output/labels/{args.mov}/seg'
    if os.path.exists(text_outdir): shutil.rmtree(text_outdir)
    os.makedirs(text_outdir, exist_ok=True)
    video_inpath = f'./data/mp4/{args.mov}.mp4'
    video_outpath = f'./output/track_mov/{args.mov}_seg.mp4'
    os.makedirs('/'.join(video_outpath.split('/')[:-1]), exist_ok=True)
    cap = cv2.VideoCapture(video_inpath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(video_outpath, fourcc, fps, (frame_width, frame_height))
    
    # [add] scene
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: 動画の最初のフレームを取得できませんでした")
        cap.release()
        out.release()
        exit()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_score = 1
    scene_number = 1
    scene_num = 0
    scene_size = int(1e6)
    seg_dict = {}

    # tracking
    print('mov -> ids')
    frame_count = 1
    pbar = tqdm(total=num_frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotator = Annotator(frame, line_width=2)
        results = model.track(frame, tracker=args.tracking_yaml, persist=True, classes=0, verbose=False)
        # [add] scene
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = calc_frame_diff(prev_gray, gray)
        delta_score = score / prev_score
        if delta_score > 1.4 and score > 15 and frame_count > 0: 
            scene_num += 1
            # シーン切り替えではフレームを赤くする
            # red_overlay = frame.copy()
            # red_overlay[:] = (0, 0, 255)  # 赤色 (BGR)
            # alpha = 0.5  # 透明度（0.0 〜 1.0）
            # frame = cv2.addWeighted(red_overlay, alpha, frame, 1 - alpha, 0)
        prev_gray = gray
        prev_score = score
        # 
        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for mask, track_id in zip(masks, track_ids):
                track_id_ = track_id + scene_num * scene_size
                if not track_id_ in seg_dict: seg_dict[track_id_] = len(seg_dict)
                if len(mask) == 0: continue
                # txt
                with open(os.path.join(text_outdir, f'{track_id_}.txt'), 'a') as f:
                    f.write(f'{frame_count}')
                    for x, y in mask:
                        f.write(f' {int(x)} {int(y)}')
                    f.write(f'\n')
                annotator.seg_bbox(mask=mask, mask_color=colors.__call__(seg_dict[track_id_]%19), label=str(track_id_))
        cv2.putText(frame, f"Scene {scene_num}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)
        out.write(frame)
        frame_count += 1
        pbar.update(1)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    cap.release()
    out.release()
    # cv2.destroyAllWindows() 
    
    # add audio
    video2_outpath = video_outpath.replace('.mp4', '_audio.mp4')
    clip = VideoFileClip(video_outpath)
    audio = AudioFileClip(video_inpath)
    audio = audio.set_duration(clip.duration)
    video = clip.set_audio(audio)
    video.write_videofile(video2_outpath, codec='libx264', audio_codec='aac')
    if os.path.exists(video_outpath): os.remove(video_outpath)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mov', type=str, help='動画(mp4)のパスを指定してください', required=True)
    parser.add_argument('--yolo-seg-weights', type=str, help='YOLOの重みを指定してください', required=True)
    parser.add_argument('--tracking-yaml', type=str, help='トラッキングモデル(.yaml)を指定してください', required=True)
    args = parser.parse_args()
    main(args)