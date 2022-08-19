from tqdm import tqdm

import time
import numpy as np

import torch
import torch.nn as nn

from utils.accuracy_score_compute import Evaluate
#https://blog.csdn.net/u012370185/article/details/94409933


def evaluate(model, val_dataloader, log, args):
    """Runs official COCO evaluation.
    
    limit: if not 0, it's the number of images to use for evaluation
   

    Args:
        model : The model. 
        val_dataloader : A DataLoader object with valiadtion data
        cocoGT : coco GroundTruth
        eval_type : "bbox" or "segm" for bounding box or segmentation evaluation. Defaults to "segm".
    """

    model.eval()
    
    
    t_start = time.time()
    Iou = []
    evaluate = Evaluate(args)
    for n_batch, sample in enumerate(tqdm(val_dataloader, desc='Evaluate:')):
        with torch.no_grad():
            image = sample[0]["image"].cuda()
            target = sample[0]['label'].cuda()
            origin_size = sample[1]
            image_var = torch.autograd.Variable(image).float()
            target_var = torch.autograd.Variable(target).long()

            
            prediction = model(image_var)
            prediction = nn.functional.interpolate(
            prediction, size=target_var.size()[1:], mode="bilinear", align_corners=False)
            prediction = prediction.cpu().numpy()
            target_var = target_var.cpu().numpy()
            prediction = prediction.argmax(axis=1).astype(np.uint8)
            evaluate.hist_martrix(prediction,target_var)
    acc, acc_cls, iou, fwavacc = evaluate.label_accuracy_score()
    Iou.append(iou)
       
        
    MIOU = sum(Iou)/len(Iou)  
    total_time = time.time() - t_start
    log.logger.info("MIoU is: {}. ".format(MIOU))
    log.logger.info("Total time:{}".format(float(total_time)))

