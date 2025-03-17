import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from facenet_pytorch import InceptionResnetV1

# cp ../face_recognition/weights/facenet/cutie_street_youtube_yolo_seg_live/best_val_loss_model.pt ./weights/facenet_cutie_street_youtube_yolo_seg_live.pt

class FaceNet(nn.Module):
    
    def __init__(self, output_dim, device):
        super().__init__()
        self.device = device
        model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=output_dim)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels)
        results = {
            'pred': preds,
            'loss': loss,
            'acc': acc
        }
        return results
    
    def inference(self, inputs):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        outputs = self.softmax(outputs)
        prob, pred = torch.max(outputs, 1)
        return prob, pred
    