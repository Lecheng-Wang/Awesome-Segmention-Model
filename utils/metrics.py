# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${2025/5/15} ${20:13}
# @Function: metrics in ML(machine-learning) and  DL(deep-learning)

'''
        (OA)        Overall_Accuracy、
        (Kappa)     Cohen's Kappa、
        (mIoU)      mean Intersection over Union、
        (FWIoU)     Frequency Weighted Intersection over Union
        (Precision) Precision
        (Recall)    Recall
        (F1 score)  F1 score
        (F2 score)  F2 score
'''


import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class        = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    # (OA) Overall_Accuracy
    def OverAll_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    # (Kappa) Cohen's Kappa
    def Kappa(self):
        N       = self.confusion_matrix.sum()
        if N == 0:
            return 0.0
        Po       = np.diag(self.confusion_matrix).sum() / N
        row_sums = self.confusion_matrix.sum(axis=1)
        col_sums = self.confusion_matrix.sum(axis=0)
        Pe       = np.sum(row_sums * col_sums) / (N ** 2)
        if Pe == 1:
            return 1.0
        kappa    = (Po - Pe) / (1 - Pe + 1e-8)
        return kappa

    # (mIoU) mean Intersection over Union
    def mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(IoU)
        return MIoU,IoU

    # (FWIoU) Frequency Weighted Intersection over Union
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        iu   = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            np.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # (Precision) Precision
    def Precision(self):
        tp             = np.diag(self.confusion_matrix)
        pred_per_class = self.confusion_matrix.sum(axis=0)
        precision      = np.divide(tp, pred_per_class, out=np.zeros_like(tp), where=(pred_per_class != 0))
        mean_precision = np.nanmean(precision)
        return mean_precision,precision
    
    # (Recall) Recall
    def Recall(self):
        tp               = np.diag(self.confusion_matrix)
        actual_per_class = self.confusion_matrix.sum(axis=1)
        recall           = np.divide(tp, actual_per_class, out=np.zeros_like(tp), where=(actual_per_class != 0))
        mean_recall      = np.nanmean(recall)
        return mean_recall,recall
    
    # (F1 score) F1 score
    def F1_Score(self):
        tp               = np.diag(self.confusion_matrix)
        pred_per_class   = self.confusion_matrix.sum(axis=0)
        actual_per_class = self.confusion_matrix.sum(axis=1)
        precision        = np.divide(tp, pred_per_class, out=np.zeros_like(tp), where=(pred_per_class != 0))
        recall           = np.divide(tp, actual_per_class, out=np.zeros_like(tp), where=(actual_per_class != 0))
        f1_scores        = 2 * (precision * recall) / (precision + recall + 1e-8)
        mean_f1_score    = np.nanmean(f1_scores)
        return mean_f1_score,f1_scores

    # (F2 score) F2 score
    def F2_Score(self):
        tp               = np.diag(self.confusion_matrix)
        pred_per_class   = self.confusion_matrix.sum(axis=0)
        actual_per_class = self.confusion_matrix.sum(axis=1)
        precision        = np.divide(tp, pred_per_class, out=np.zeros_like(tp), where=(pred_per_class != 0))
        recall           = np.divide(tp, actual_per_class, out=np.zeros_like(tp), where=(actual_per_class != 0))
        f2_scores        = (5 * precision * recall) / (4 * precision + recall + 1e-8)
        mean_f2_score    = np.nanmean(f2_scores)
        return mean_f2_score,f2_scores

    # Generate Confusion Matrix
    def _generate_matrix(self, gt_image, pre_image):
        mask  = (gt_image >= 0) & (gt_image < self.num_class) & (pre_image >= 0) & (pre_image < self.num_class)
        label = self.num_class * gt_image[mask].astype(int) + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        return count.reshape(self.num_class, self.num_class)
    
    # Update Confusion Matrix per-batch
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
    
    # Reset Confusion Matrix
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)