# Thanks to  :)  https://github.com/wkentaro/pytorch-fcn
import numpy as np


def _fast_hist(pred, gt, num_class):
    """ fast_hist. 

    Args:
        gt : ground truth label. [HxW,]
        pred : prediction. [HxW,]
        num_class : Contains background class. coco: 81; VOC: 21

    Returns:
        混淆矩阵  https://blog.csdn.net/u012370185/article/details/94409933
    """
    k = (gt >= 0) & (gt < num_class)#k is bool，Shape (H×W,)；目的是找出标签中需要计算的类别（去掉了背景）
    #np.bincount计算了从0到num_classes**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    return np.bincount(num_class * gt[k].astype(int) + pred[k], minlength=num_class ** 2).reshape(num_class, num_class)
 

class Evaluate(object):
    """Compute MIoU, accuracy score.
    Return:
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
    """
    def __init__(self, args):
        self.n_class  = args.data.NUM_CLASSES
        self.args = args
        self.hist = np.zeros((self.n_class, self.n_class))
    def hist_martrix(self, pred, gt):
        """ 
        Args:
            pred : prediction . [batch_size, num_classes, height, width]
            gt: ground truth label. [batch_size, height, width]    
        """
        for pred_label, gt_label in zip(pred, gt):
            self.hist += _fast_hist(pred_label.flatten(), gt_label.flatten(), self.n_class)
    def label_accuracy_score(self):

        acc = np.diag(self.hist).sum() / self.hist.sum()
        #with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        #with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(self.hist) / (
            self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist)
        )
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc

