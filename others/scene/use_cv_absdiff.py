import os
import cv2
import subprocess
import numpy as np
import matplotlib.pyplot as plt


# python others/scene/use_cv_absdiff.py
def calc_frame_diff(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    diff_mean = np.mean(diff)
    return diff_mean


def main():
    name = "kawadame_ebisulive.mp4"
    # name = "kawadame.mp4"
    video_path = f"data/mp4/{name}"
    output_dir = "output/split"
    output_path = os.path.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    # 動画読み込み
    cap = cv2.VideoCapture(video_path)

    # 動画情報取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 動画保存設定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 初期設定
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: 動画の最初のフレームを取得できませんでした")
        cap.release()
        out.release()
        exit()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_score = 1
    scene_number = 1  # シーン番号
    scores = []
    delta_scores = []
    scene_changes = [0]
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame diff
        score = calc_frame_diff(prev_gray, gray)
        delta_score = score / prev_score
        # judge
        # if score > 30: scene_number += 1
        if delta_score > 1.5 and score > 10 and frame_num > 0: 
            scene_number += 1
            scene_changes.append(frame_num)
            color = (0, 0, 255)
        else:
            color = (128, 128, 128)
        cv2.putText(frame, f"Scene {scene_number}, Score {score}, Delta {delta_score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        out.write(frame)
        prev_gray = gray
        prev_score = score
        scores.append(score)
        delta_scores.append(delta_score)
        frame_num += 1
    scene_changes.append(frame_num)
    cap.release()
    out.release()
    print("動画処理が完了しました！")
    
    
    # plt
    plt.figure(figsize=(8, 4))
    time = range(len(scores))
    plt.plot(time, scores, linestyle='-', label="score", color='b')
    plt.plot(time, delta_scores, linestyle='--', label="diff_score", color='r')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'img.png'))
    
    
    # ** ffmpeg を用いて各シーンを動画として分割保存 **
    # output_scene_dir = os.path.join(output_dir, 'scene')
    # os.makedirs(output_scene_dir, exist_ok=True)
    # for i in range(len(scene_changes) - 1):
    #     start_frame = scene_changes[i]
    #     end_frame = scene_changes[i + 1]
    #     start_time = start_frame / fps
    #     duration = (end_frame - start_frame) / fps
    #     output_file = os.path.join(output_scene_dir, f"scene_{i + 1}.mp4")
    #     command = [
    #         "ffmpeg",
    #         "-i", video_path,                # 元の動画
    #         "-ss", str(start_time),          # 開始時間
    #         "-t", str(duration),             # 切り取り時間
    #         "-c", "copy",                    # 再エンコードなしで映像・音声をコピー
    #         output_file
    #     ]
    #     print(f"Extracting {output_file}...")
    #     subprocess.run(command)


if __name__ == '__main__':
    main()