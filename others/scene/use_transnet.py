import cv2
import numpy as np
import tensorflow as tf


# python others/use_transnet.py
class TransNetV2:
    """TransNetV2 モデルを使用したシーン切り替え検出"""
    def __init__(self, model_path="TransNetV2.pb"):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """モデルをロード"""
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session()
            with tf.io.gfile.GFile(self.model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

            self.input_tensor = self.graph.get_tensor_by_name("input_1:0")
            self.output_tensor = self.graph.get_tensor_by_name("sigmoid_output:0")

    def predict(self, frames):
        """動画フレームのリストを入力してシーン変化スコアを取得"""
        frames = np.array(frames).astype(np.float32) / 255.0
        frames = np.expand_dims(frames, axis=0)  # バッチ次元を追加
        return self.sess.run(self.output_tensor, feed_dict={self.input_tensor: frames})[0]


def main():
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    transnet = TransNetV2()
    frames = []
    scene_changes = []
    frame_idx = 0
    BATCH_SIZE = 25  # TransNetV2 の入力サイズ

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (48, 27))  # TransNetV2 の入力サイズにリサイズ
        frames.append(frame)
        if len(frames) == BATCH_SIZE:
            scores = transnet.predict(frames)
            for i, score in enumerate(scores):
                if score > 0.5:  # スコアが高ければカット判定
                    print(f"Scene change detected at frame {frame_idx - BATCH_SIZE + i}")
                    scene_changes.append(frame_idx - BATCH_SIZE + i)
            frames.pop(0)  # FIFOでバッファを維持
        frame_idx += 1
    cap.release()
    print("Detected scene changes:", scene_changes)

if __name__ == '__main__':
    main()