from torch import log_softmax
import torch.nn as nn
class Segmentation_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.LogSoftmax(dim=1)
        #self.layer = nn.Softmax(dim=1)
        self.loss = nn.NLLLoss(ignore_index=-1)#size_average=False
        self.loss1 = nn.CrossEntropyLoss(ignore_index=-1)
    def forward(self, seg_predict, seg_target):
        """To compute the semantic segmentation loss.

        Args:
            seg_predict (batch_size, num_classes, h, w): The prediction out.
            seg_target (batch_size,  h, w): The ground truth.

        Returns:
            total loss.
        """
        loss = self.loss(self.layer1(seg_predict),seg_target)
       
        return loss 