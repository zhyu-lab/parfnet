import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1)
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            loss = 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            loss = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                loss += 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
            loss /= self.num_classes
        return loss

class DiceBceLoss(nn.Module):
    def __init__(self, num_classes=1, bce_weight=0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        target = target.squeeze(1)
        dice_loss = 0
        bce_loss = 0
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)

            bce_loss = nn.BCEWithLogitsLoss()(pred, target)
        else:
            pred = F.softmax(pred, dim=1)
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice_loss += 1 - (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss /= self.num_classes

            bce_loss = nn.BCEWithLogitsLoss()(pred, target.unsqueeze(1))

        total_loss = (1 - self.bce_weight) * dice_loss + self.bce_weight * bce_loss
        return total_loss


class BceLoss(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        target = target.unsqueeze(1)

        if self.num_classes == 1:
            loss = nn.BCEWithLogitsLoss()(pred.view(-1), target.view(-1))
        else:
            pred = F.softmax(pred, dim=1)
            loss = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                loss += nn.BCEWithLogitsLoss()(pred_c, target_c)
            loss /= self.num_classes

        return  loss