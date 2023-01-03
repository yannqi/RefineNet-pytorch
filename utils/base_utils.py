import os
import numpy as np 
import torch
import torch.distributed as dist
def init_distributed_mode(args):
    # set up distributed device
    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    args.device = torch.device("cuda", args.local_rank)
    print(args.device,'argsdevice')
    args.world_size = torch.distributed.get_world_size()
    print(f"[init] == local rank: {args.local_rank}, global rank: {args.rank} ==")
    
def set_random_seed(seed):
    """Set random seed."""    
    np.random.seed(seed)
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True   
        torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def eta_compute(time_per_epoch,current_epoch,epochs):
    """Estimated Time of Arrival. Return the string %H

    Args:
        time_per_epoch (float): Running time per epoch.
        current_epoch (int): The current epoch.
        epochs (int): Total epochs.
    """
    eta = time_per_epoch * (epochs-current_epoch)/3600
    return eta


class AverageMeter(object):
    """AverageMeter可以记录当前的输出,并累加到之前的输出之中,然后根据需要可以打印出历史上的平均值。
    call: obj = AverageMeter()
    reset object : obj.reset()
    undate object: obj.update(x)
    get average: obj.avg
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k/ batch_size)
    return res