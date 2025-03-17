import os
import sys
sys.path.append(os.pardir)
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from math import atan2, degrees
from collections import Counter
from facenet_pytorch import MTCNN, fixed_image_standardization
from torchvision import transforms
from face_recognition.model import FaceNet


# python bin/id2name.py --mov 61thSingleDance_60_85_1k --member-list face_recognition/member_list/61thsingle.txt --facenet-weights weights/facenet_61thsingle.pt --device cuda
MINSIZE = 150


# transform = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])


def cut_img(image, seg_mask):
    seg_mask = np.array(seg_mask, dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [seg_mask], (255, 255, 255))
    masked_image = cv2.bitwise_and(image, mask)
    x, y, w, h = cv2.boundingRect(seg_mask)
    cropped_image = masked_image[y:y+h, x:x+w]
    # cv2.imwrite(f'img/{x}.jpg', image[y:y+h, x:x+w])
    # cv2.imwrite(f'img/mask.jpg', cropped_image)
    return cropped_image


def is_face_facing_forward(landmarks, threshold=15):
    # 左目と右目の中心を計算
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    # 目の中心点のX座標の差とY座標の差を求める
    diff_x = right_eye[0] - left_eye[0]
    diff_y = right_eye[1] - left_eye[1]
    # 目の傾き角度を求める
    angle = degrees(atan2(diff_y, diff_x))
    # 正面を向いていると判断する角度の範囲
    return abs(angle) <= threshold


def main(args):
    
    # path
    video_path = f'data/mp4/{args.mov}.mp4'
    text_dir = f'output/labels/{args.mov}/seg'
    os.makedirs(text_dir, exist_ok=True)
    
    # video data
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # class
    with open(args.member_list, 'r', encoding='utf-8') as f:
        classes = f.read().splitlines()
    classes.append('nan')
    num_classes = len(classes)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # model
    device = torch.device(args.device)
    # yolo-face model
    yolo_face = YOLO(args.yolo_face_weights)
    # mtcnn
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=MINSIZE,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    # facenet
    facenet = FaceNet(num_classes-1, device)
    print(args.facenet_weights)
    facenet.load_state_dict(torch.load(args.facenet_weights, map_location=device))
    facenet.to(device)
    facenet.eval()
    
    # tracking data
    idxs = os.listdir(text_dir)
    idxs = [int(idx.replace('.txt', '')) for idx in idxs if '.txt' in idx]
    ts = [[] for _ in range(num_frame+1)]
    track_datas = []
    for idx in idxs:
        with open(os.path.join(text_dir, f'{idx}.txt'), 'r', encoding='utf-8') as f:
            track_data = [[int(value) for value in line.split(' ')] for line in f]
            for data in track_data:
                t = data[0]
                seg_points = data[1:]
                mask = [seg_points[i:i+2] for i in range(0, len(seg_points), 2)]
                ts[t].append([idx, mask])
    idxs = sorted(idxs)
    
    # 顔識別
    print('id -> name')
    frame_count = 0
    pbar = tqdm(total=num_frame)
    frame_size = max(frame_width, frame_height)
    outputs = [[] for _ in range(idxs[-1]+1)]
    all_probs = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # その時間のトラッキングの結果を mtcnn -> facenet
        for idx, seg_mask in ts[frame_count]:
            human_image, (x, y) = cut_img(frame, seg_mask)
            box_and_conf = yolo_face.predict(human_image, verbose=False)[0].boxes
            yolo_boxes = box_and_conf.xyxy.tolist()
            if len(yolo_boxes) != 1: continue
            f_x1, f_y1, f_x2, f_y2 = map(int, yolo_boxes[0])
            if max(f_x2-f_x1, f_y2-f_y1) < MINSIZE: continue
            human_image = human_image[..., ::-1]
            face_image = Image.fromarray(human_image[int(f_y1):int(f_y2), int(f_x1):int(f_x2)])
            face_image = transform(face_image)
            with torch.no_grad():
                prob, pred_idx = facenet.inference(face_image.unsqueeze(0))
                prob, pred_idx = prob.item(), pred_idx.item()
            outputs[idx].append([pred_idx, prob, frame_count])
            all_probs.append(prob)
        frame_count += 1
        pbar.update(1)
    cap.release()

    # FaceNet結果 -> 名前
    names = {}
    idx_max = max(idxs)
    for idx in idxs:
        if len(outputs[idx]) > 0:
            sorted_output = sorted(outputs[idx], key=lambda x: x[1])
            list_class = [output[0] for output in sorted_output]
            list_prob = [output[1] for output in sorted_output]
            counter = Counter(list_class)
            most_common_class, count = counter.most_common(1)[0]
            if count/len(list_class) >= 0.5:
                names[idx] = idx_to_class[most_common_class]
            else:
                names[idx] = 'nan'
        else:
            names[idx] = 'nan'

    # output
    output_path = os.path.join('/'.join(text_dir.split('/')[:-1]), 'id_name.txt')
    if os.path.exists(output_path): os.remove(output_path)
    sorted_names = dict(sorted(names.items()))
    with open(output_path, 'a') as f:
        for key, value in sorted_names.items():
            print(key, value)
            f.write(f'{value}\n')           
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mov', type=str, help='動画(mp4)のパスを指定してください', required=True)
    parser.add_argument('--member-list', type=str, help='メンバーリスト(.txt)のパスを指定してください', required=True)
    parser.add_argument('--yolo-face-weights', type=str, help='YOLOの重みを指定してください', required=True)
    parser.add_argument('--facenet-weights', type=str, help='FaceNetの重みを指定してください', required=True)
    parser.add_argument('--device', type=str, help='cuda device or cpu', required=True)
    args = parser.parse_args()
    main(args)