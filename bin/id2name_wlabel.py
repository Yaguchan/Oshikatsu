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


# python bin/id2name_wlabel.py --mov 61thSingleDance_60_85_1k --member-list face_recognition/member_list/61thsingle.txt --facenet-weights weights/facenet_61thsingle.pt --device cuda
# THRES = 0.99
MINSIZE = 150
STRIDE = 5


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
    return cropped_image, (x, y)


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


def get_face_xyxy(seg_mask, xyxy):
    f_x1, f_y1, f_x2, f_y2 = xyxy
    x, y = seg_mask[0][0], seg_mask[0][1]
    return (f_x1+x, f_y1+y, f_x2+x, f_y2+y)


def draw_mask_and_label(frame, seg_mask, label, color, bbox):
    seg_mask = np.array(seg_mask, dtype=np.int32)
    cv2.polylines(frame, [seg_mask], isClosed=True, color=color, thickness=4)
    x, y, w, h = cv2.boundingRect(seg_mask)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    f_x1, f_y1, f_x2, f_y2 = bbox
    cv2.rectangle(frame, (f_x1, f_y1), (f_x2, f_y2), color, 4)
    

def main(args):
    
    # path
    video_path = f'data/mp4/{args.mov}.mp4'
    text_dir = f'output/labels/{args.mov}/seg'
    output_video_path = f'output/labels/{args.mov}/wlabel.mp4'
    os.makedirs(text_dir, exist_ok=True)
    
    # video data
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # class
    with open(args.member_list, 'r', encoding='utf-8') as f:
        classes = f.read().splitlines()
    classes.append('nan')
    num_classes = len(classes)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    colors = [(0, 0, 255), (255, 0, 0), (0, 165, 255), (203, 192, 255), (0, 255, 0), (255, 255, 0), (211, 0, 148), (0, 255, 255)]
    
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
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(f"img/slides/{idx}_{frame_count}_frame.jpg")
            Image.fromarray(cv2.cvtColor(human_image, cv2.COLOR_BGR2RGB)).save(f"img/slides/{idx}_{frame_count}_human.jpg")
            # if human_image.shape[0] < MINSIZE or human_image.shape[1] < MINSIZE: continue
            # xyxy_list, _, landmarks = mtcnn.detect(human_image, landmarks=True)
            # if xyxy_list is None: continue
            # if len(xyxy_list) != 1: continue
            # f_x1, f_y1, f_x2, f_y2 = map(int, xyxy_list[0])
            box_and_conf = yolo_face.predict(human_image, verbose=False)[0].boxes
            yolo_boxes = box_and_conf.xyxy.tolist()
            if len(yolo_boxes) != 1: continue
            f_x1, f_y1, f_x2, f_y2 = map(int, yolo_boxes[0])
            if max(f_x2-f_x1, f_y2-f_y1) < MINSIZE: continue
            # if not is_face_facing_forward(landmarks[0]): continue
            human_image = human_image[..., ::-1]
            face_image = Image.fromarray(human_image[int(f_y1):int(f_y2), int(f_x1):int(f_x2)])
            face_image.save(f"img/slides/{idx}_{frame_count}_humanface.jpg")
            # face_image.save(f"img/{idx}.jpg")
            face_image = transform(face_image)
            with torch.no_grad():
                prob, pred_idx = facenet.inference(face_image.unsqueeze(0))
                prob, pred_idx = prob.item(), pred_idx.item()
            # 足切り（FaceNet出力確率）
            # if prob < THRES: 
            #     continue
            outputs[idx].append([pred_idx, prob, frame_count])
            all_probs.append(prob)
            label = f'ID: {pred_idx} ({prob:.2f})'
            draw_mask_and_label(frame, seg_mask, label, colors[pred_idx], (f_x1+x, f_y1+y, f_x2+x, f_y2+y))
        out.write(frame)
        frame_count += 1
        pbar.update(1)
    cap.release()
    out.release()

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
            # # class
            # list_class = [output[0] for output in outputs[idx]]
            # # frame_count
            # list_frame_count = [output[2] for output in outputs[idx]]
            # list_final_class = []
            # list_change_time = []
            # if len(outputs[idx]) > STRIDE:
            #     b_cls = max(list_class[0:STRIDE], key=list_class.count)
            #     list_final_class.append(b_cls)
            #     for i in range(11, len(list_frame_count)):
            #         counter = Counter(list_class[i-STRIDE:i])
            #         n_cls, count = counter.most_common(1)[0]
            #         if b_cls != n_cls and count > (STRIDE // 2):
            #             list_final_class.append(n_cls)
            #             list_change_time.append(list_frame_count[(i-STRIDE)+list_class[i-STRIDE:i].index(n_cls)])
            #             b_cls = n_cls
            # else:
            #     b_cls = max(list_class, key=list_class.count)
            #     list_final_class.append(b_cls)
            # with open(os.path.join(text_dir, f'{idx}.txt'), 'r', encoding='utf-8') as f:
            #     track_datas = [line for line in f]
            #     ts = [int(track_data.split(' ')[0]) for track_data in track_datas]
            # indices = [ts.index(x) for x in list_change_time]
            # list_track_data = [track_datas[i:j] for i, j in zip([0] + indices, indices + [len(track_datas)])]
            # idx_max = idx_max + len(list_track_data) - 1
            # list_idx = [idx] + [i for i in range(idx_max-len(list_track_data)+2, idx_max+1)]
            # for idx, final_class, track_data in zip(list_idx, list_final_class, list_track_data):
            #     names[idx] = idx_to_class[final_class]
            #     idx_path = os.path.join(text_dir, f'{idx}.txt')
            #     if os.path.exists(idx_path): os.remove(idx_path)
            #     with open(idx_path, 'a', encoding='utf-8') as f:
            #         for data in track_data:
            #             f.write(data)
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