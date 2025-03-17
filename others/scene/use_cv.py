import os
import cv2
import numpy as np


# python others/scene/use_cv.py
def calc_frame_diff(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    diff_mean = np.mean(diff)
    return diff_mean


def calc_hist_diff(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def main():
    # name = "kawadame_ebisulive.mp4"
    name = "kawadame.mp4"
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
    scene1_number = 1  # シーン番号
    scene2_number = 1  # シーン番号

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame diff
        score1 = calc_frame_diff(prev_gray, gray)
        if score1 > 30: scene1_number += 1
        # hist diff
        score2 = calc_hist_diff(prev_gray, gray)
        if score2 > 0.2: scene2_number += 1
        cv2.putText(frame, f"Scene {scene1_number}, Score {score1}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Scene {scene2_number}, Score {score2}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        prev_gray = gray
    cap.release()
    out.release()
    print("動画処理が完了しました！")


if __name__ == '__main__':
    main()