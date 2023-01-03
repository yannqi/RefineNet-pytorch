import torch
import time
import torch.nn as nn 
from utils.base_utils import  AverageMeter
def train_loop(train_dataloader, model, criterion, optimizer, log, args, epoch,scaler):
    """Traditional dataloader style."""
    model.train()
    losses = AverageMeter()
    for n_batch, sample in enumerate(train_dataloader):
        image = sample["image"].cuda()
        target = sample['label'].cuda()
        image_var = torch.autograd.Variable(image).float()
        target_var = torch.autograd.Variable(target).long()

        with torch.cuda.amp.autocast(enabled=args.amp):  # Automatic Mixed Precision
            
            prediction = model(image_var)
            prediction = nn.functional.interpolate(
            prediction, size=target_var.size()[1:], mode="bilinear", align_corners=False)
            loss = criterion(prediction,target_var)
            
        if args.warmup is not None:
            warmup(optimizer, args.warmup, epoch, n_batch, args.learning_rate)
            
         # scaler Automatic Mixed Precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        losses.update(loss.item(), image.size(0))
        if args.local_rank == 0 and n_batch % args.report_period == 0:
            log.logger.info(
                "epoch: %d , iter: %d/%d , ,loss: %.4f"
                % (epoch + 1, n_batch + 1, len(train_dataloader), loss.item())
            )    
    
    return losses.avg



def warmup(optim, warmup_iters, epoch, n_batch, base_lr):
    if epoch ==1 and n_batch < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * n_batch
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr





def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]
    
    
    