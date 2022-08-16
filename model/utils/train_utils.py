from torch.autograd import Variable
import torch
import time
import torch.nn as nn 
from apex import amp




def train_loop(model, criterion, scaler, epoch, optim, train_dataloader, num_data, args,log):
    """Traditional dataloader style."""
    total_loss = 0
    step_start = time.time()
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
            total_loss += loss
       
            

        if args.warmup is not None:
            warmup(optim, args.warmup, epoch, n_batch, args.learning_rate)
            
         # scaler Automatic Mixed Precision
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        
        if n_batch % args.report_period == 0 :  
            step_stop = time.time()
            use_time = int(step_stop - step_start)
            mean_loss =  total_loss/n_batch
            step_start = time.time()
            
            if args.local_rank == 0:
                log.logger.info('-TRAINED: {0:10d}/{1} \n - USED: {2} s \n- loss: {3:.4f}  - mean loss: {4:.4f}'
                .format((n_batch + 1) * args.batch_size,
                        num_data, use_time,loss,mean_loss))
    mean_loss =  total_loss/num_data  
    if args.local_rank == 0:   
        log.logger.info('Epoch: %s, loss: %s',epoch, mean_loss.item())
    return mean_loss



def warmup(optim, warmup_iters, epoch, n_batch, base_lr):
    if epoch ==1 and n_batch < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * n_batch
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def load_checkpoint(model, checkpoint):
    """Load model from checkpoint."""
    
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)


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
    
    
    