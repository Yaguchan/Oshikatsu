# 0.設定
MOVNAME="kawadame_0_27"                                                        # movie name (data/mp4/MOVNAME.mp4)
YOLO_SEG_WEIGHTS="weights/yolo11x-seg.pt"                                       # YOLOv11 weights (yolo11X-seg.pt)
YOLO_FACE_WEIGHTS="weights/yolov11l-face.pt"                                    # YOLOv11 weights (yolov11X-face.pt)
FACENET_WEIGHTS="weights/facenet_cutie_street_youtube_yolo_seg_val4acc.pt"      # FaceNet weights (facenet_X.pt)
TRACKING_YAML="data/yaml/bytetrack.yaml"                                        # Tracking Model (botsort/bytetrack.yaml)
MEMBER_LIST="face_recognition/member_list/cutie_street.txt"                     # member list
MEMBER_ENJP_LIST="face_recognition/member_list/member_cutie_street.csv"         # member en/jp list
MEMBER_NAME="RisaFurusawa"                                                    # member name
FONT="data/font/NotoSansJP-Black.ttf"                                           # Font
DEVICE="cuda:0"                                                                 # cuda or mps or cpu

# 実行　bash run/run_seg.sh
. path.sh
# 1. トラッキング
python bin/mov2ids_scene.py --mov $MOVNAME --yolo-seg-weights $YOLO_SEG_WEIGHTS --tracking-yaml $TRACKING_YAML
# 2. 顔識別
python bin/id2name.py --mov $MOVNAME --member-list $MEMBER_LIST --yolo-face-weights $YOLO_FACE_WEIGHTS --facenet-weights $FACENET_WEIGHTS --device $DEVICE
# 3. 切り抜き作成
python bin/make_short.py --mov $MOVNAME --name $MEMBER_NAME
# (option) 動画作成
# python bin/make_mov.py --mov $MOVNAME --member-enjp-list $MEMBER_ENJP_LIST --font $FONT