import os
import cv2
import shutil
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict
from moviepy.editor import VideoFileClip, AudioFileClip
from ultralytics.utils.plotting import Annotator, colors


# python bin/mov2ids.py --mov 62thSingleDance_178_182_1k --yolo-seg-weights weights/yolo11x-seg.pt --tracking-yaml data/yaml/bytetrack.yaml


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

    # tracking
    print('mov -> ids')
    frame_count = 0
    pbar = tqdm(total=num_frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotator = Annotator(frame, line_width=2) # anaconda/anaconda3/envs/oshikatsu_/lib/python3.10/site-packages/ultralytics/utils/plotting.py
        results = model.track(frame, tracker=args.tracking_yaml, persist=True, classes=0, verbose=False)
        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for mask, track_id in zip(masks, track_ids):
                if len(mask) == 0: continue
                # txt
                with open(os.path.join(text_outdir, f'{track_id}.txt'), 'a') as f:
                    f.write(f'{frame_count}')
                    for x, y in mask:
                        f.write(f' {int(x)} {int(y)}')
                    f.write(f'\n')
                annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=str(track_id))
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