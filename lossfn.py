
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import Facemask_utils

class FacemaskLoss(nn.Module):
    def __init__(self):
        super(BasketLoss, self).__init__()
        self.neg_pos_ratio = 10
    def forward(self,scores, predicted_locations, labels, gt_locations,not_ignored):
        num_classes = scores.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(scores, dim=2)[:, :, 0]
            mask = basket_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio).int() & not_ignored.int()

        confidence = scores[mask.long(), :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        mse_loss = F.mse_loss(predicted_locations, gt_locations, reduction='sum')
        #smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return mse_loss / num_pos, classification_loss / num_pos
