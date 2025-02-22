# Oshikatsu
動画からショート動画の作成を行うツールです。
![Image](https://github.com/user-attachments/assets/6fc71e67-caf2-405a-9a5c-ef1a5c6e659e)
1. 顔識別モデルの学習

2. 動画の中に出現するメンバーのトラッキング情報を獲得し、メンバーごとの動画における出現
![Image](https://github.com/user-attachments/assets/d52ef743-e9e1-4609-8ec6-deb9f0c22bdd)



## 実行
### 環境構築
必要なものを以下URLからダウンロードしてください
<details><summary>ダウンロード</summary>

・[YOLOv8 weights](https://github.com/ultralytics/ultralytics)  
・[Tracking Model](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)

</details>

```
conda env create -f env.yaml
```

### 推論
設定の項目を埋めて実行
<details><summary>設定項目</summary>

・`MOVNAME`             ：movie name (data/mp4/MOVNAME.mp4)  
・`YOLO_WEIGHTS`        ：YOLOv8 weights (yolov8X.pt)  
・`FACENET_WEIGHTS`     ：FaceNet weights (facenet_X.pt)  
・`TRACKING_YAML`       ：Tracking Model (botsort or bytetrack.yaml)  
・`MEMBER_LIST`         ：メンバーのリスト  
・`MEMBER_ENJP_LIST`    ：メンバーの名前の日本語/英語データ  
・`FONT_PATH`           ：使用するフォント  
・`DEVICE`              ：cuda or mps or cpu  

</details>

セグメンテーション無しver
```
bash run/run.sh
```
セグメンテーション有りver
```
bash run/run_seg.sh
```