
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchmetrics

def plot(a):
    plt.figure()
    plt.imshow(a)


class DiceScore(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1).float()
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = F.one_hot(pred, self.num_classes).permute(0, 3, 1, 2).float()
            dice = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice += (2. * intersection + self.smooth) / (union + self.smooth)
            dice /= self.num_classes
        return dice

class IoU(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1).float()
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            den = union - intersection
            iou = (intersection + self.smooth) / (den + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = F.one_hot(pred, self.num_classes).permute(0, 3, 1, 2).float()
            iou = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                den = union - intersection
                iou += (intersection + self.smooth) / (den + self.smooth)
            iou /= self.num_classes
        return iou


class Accuracy(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.acc = torchmetrics.Accuracy(task="binary").cuda()

    def forward(self, pred, target):
        pred = pred.squeeze(1).float()
        target = target.squeeze(1).float()

        acc_value = self.acc(pred, target)
        return acc_value

class F1Score(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.f1 = torchmetrics.F1Score(task="binary").cuda()


    def forward(self, pred, target):
        pred = pred.squeeze(1).float()
        target = target.squeeze(1).float()


        f1 = self.f1(pred, target)
        return f1

class PrecisionMetric(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.precision = torchmetrics.Precision(task="binary").cuda()


    def forward(self, pred, target):
        pred = pred.squeeze(1).float()
        target = target.squeeze(1).float()

        precision_value = self.precision(pred, target)
        return precision_value

class RecallMetric(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.precision = torchmetrics.Recall(task="binary").cuda()

    def forward(self, pred, target):
        pred = pred.squeeze(1).float()
        target = target.squeeze(1).float()
        precision = self.precision(pred, target)
        return precision




class MetricWrapper(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.acc = torchmetrics.Accuracy(task="binary").cuda()
        self.f1 = torchmetrics.F1Score(task="binary").cuda()
        self.precision = torchmetrics.Precision(task="binary").cuda()
        self.recall = torchmetrics.Recall(task="binary").cuda()
        self.auc = torchmetrics.AUROC(task="binary").cuda()

    def forward(self, pred, target):
        pred = pred.squeeze(1).float()
        target = target.squeeze(1).float()

        # 计算各类指标
        acc_value = self.acc(pred, target)
        f1_value = self.f1(pred, target)
        precision_value = self.precision(pred, target)
        recall_value = self.recall(pred, target)
        auc_value = self.auc(pred, target)

        return {
            'accuracy': acc_value,
            'f1score': f1_value,
            'precision': precision_value,
            'recall': recall_value,
            'AUC': auc_value,
        }
