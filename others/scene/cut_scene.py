import os
import cv2
import numpy as np


# python others/scene/cut_scene.py
NAME = 'kawadame_ebisulive'
VIDEO_PATH = f'data/mp4/{NAME}.mp4'
OUTPUTDIR = f'output/scene/{NAME}' 


def calc_frame_diff(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    diff_mean = np.mean(diff)
    return diff_mean


def remove_close_elements(lst, threshold=15):
    result = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - result[-1] > threshold:
            result.append(lst[i])
    return result



def main():
    os.makedirs(OUTPUTDIR, exist_ok=True)

    # 動画読み込み
    cap = cv2.VideoCapture(VIDEO_PATH)

    # 動画情報取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 初期設定
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: 動画の最初のフレームを取得できませんでした")
        cap.release()
        exit()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    list_scene = [0]
    frame_count = 0
    prev_score = 1
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = calc_frame_diff(prev_gray, gray)
        delta_score = score / prev_score
        if delta_score > 1.4 and score > 15 and frame_count > 0: 
            list_scene.append(frame_count+1)
        prev_gray = gray
        prev_score = score
        frame_count += 1
    list_scene.append(total_frames)
    list_scene = remove_close_elements(list_scene)

    for i in range(len(list_scene) - 1):
        temp_video_path = os.path.join(OUTPUTDIR, f'temp_clip_{i}.mp4')
        temp_audio_path = os.path.join(OUTPUTDIR, f'temp_clip_{i}.aac')
        final_clip_path = os.path.join(OUTPUTDIR, f'clip_{i}_audio.mp4')
        start, end = list_scene[i], list_scene[i+1] - 1
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_idx in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        start_time = start / fps
        end_time = end / fps
        os.system(f'ffmpeg -i {VIDEO_PATH} -ss {start_time} -to {end_time} -q:a 0 -map a {temp_audio_path}')
        os.system(f'ffmpeg -y -i {temp_video_path} -i {temp_audio_path} -c:v copy -c:a aac -strict experimental {final_clip_path}')
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
    cap.release()


if __name__ == '__main__':
    main()