import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
import model.utils.seg_metrics as seg_metrics
import utils.base_utils as base_utils
import data.make_dataloader as make_dataloader

from model.refinenet import rf101, rf50
from model.utils.loss import Segmentation_Loss
from model.utils.train_utils import  tencent_trick, train_loop
from utils.model_load_save import save_checkpoint, load_checkpoint
from utils.Logger import Logger



def main():
    parser = argparse.ArgumentParser(description='RefineNet Training With PyTorch')
    parser.add_argument('--model_name', default='RefineNet', type=str,
                        help='The model name')
    parser.add_argument('--data_config', default='configs/VOC.yaml', 
                        metavar='FILE', help='path to data cfg file', type=str,)
    parser.add_argument('--device_gpu', default='2', type=str,
                        help='Cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--save_step', default=20, type=int, help='Save checkpoint every save_step')
    parser.add_argument("--resume", type=str, default=None, help="resume path")


    parser.add_argument(
        "--save",
        type=str,
        default="/data/yangqi/checkpoints_save/refine_net/checkpoints/",
        help="Save model checkpoints in the specified directory",
    )
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='number of epochs for training') #default 65
   
    parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54],
                        help='epochs at which to decay learning rate')
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--seed', '-s', default = 42 , type=int, help='manually set random seed for torch')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default= 1e-4  ,
                        help='learning rate for SGD optimizer')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum argument for SGD optimizer')
    parser.add_argument('--weight_decay', '--wd', type=float, default=5e-4,
                        help='weight-decay for SGD optimizer')
    parser.add_argument('--batch_size', '--bs', type=int, default=12, 
                        help='number of examples for each iteration')
    parser.add_argument("--base_batch_size", default=16, type=int, help="Base batch size, to adjust lr")
    parser.add_argument('--num_workers', type=int, default=4) 
    
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--backbone-path', type=str, default=None,
                        help='Path to chekcpointed backbone. It should match the'
                             ' backbone model declared with the --backbone argument.'
                             ' When it is not provided, pretrained model from torchvision'
                             ' will be downloaded.')
    parser.add_argument('--report_period', type=int, default=800, help='Report the loss every X times.')
   
    
    # Dataset Aug
    parser.add_argument("--base_size", type=int, default=224, help="Base size of the image")
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size of the image.(Final size)")
    
    # Multi Gpu
    parser.add_argument('--multi_gpu', default=False, type=bool,
                        help='Whether to use multi gpu to train the model, if use multi gpu, please use by sh.')
    
    #others 
    parser.add_argument('--amp', action='store_true', default = False,
                        help='Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')

    args = parser.parse_args()
    

    # Initialize Multi GPU
    if args.multi_gpu == True:
        base_utils.init_distributed_mode(args)
    else:
        # Use Single Gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_gpu
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        args.device = device
        args.world_size = 1
        args.local_rank = 0
    args.lr = args.lr * args.world_size * (args.batch_size / args.base_batch_size)  # * base_batch_size: 12
    args.total_batch_size = args.world_size * args.batch_size
    args.num_workers = args.world_size * args.num_workers

    # Load data configs
    cfg_path = open(args.data_config)
    cfg = yaml.full_load(cfg_path)
    cfg = edict(cfg)
    args.data = cfg
    args.save = args.save + "/{}/{}-{}-{}".format(args.model_name, args.data.NAME, args.data.NUM_CLASSES,time.strftime("%Y%m%d-%H"))
    os.makedirs(args.save, exist_ok=True)
    
    #Random seed
    base_utils.set_random_seed(args.seed)
    

    # Logger
    os.makedirs("logs/{}/".format(args.model_name), exist_ok=True)
    log_path = "{}_{}_{}".format(args.model_name, args.data.NAME, time.strftime("%Y%m%d-%H"),)
    log = Logger("logs/{}/".format(args.model_name) + log_path + ".log", level="debug")
    if args.local_rank == 0:
        log.logger.info("gpu device = %s" % args.device_gpu)
        log.logger.info("args = %s", args)

    # Load amp
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        


 
    #Logger
    log_path = '{}-{}-lr-{}-{}'.format(args.model_name, args.data.NUM_CLASSES, args.lr, time.strftime('%Y%m%d-%H'))
    
    log = Logger('logs/'+log_path+'.log',level='debug')
    
    #Initial Logging
    if args.local_rank == 0:
        log.logger.info('gpu device = %s' % args.device_gpu)
        log.logger.info('args = %s', args)

  

    #train_dataset = COCOSegmentation(args.data.DATASET_PATH, args, split='train')
    # Load dataset
    train_dataloader, val_dataloader = make_dataloader.make_dataloader(args)
    
    #Load Model 
    if args.backbone == 'resnet101' :
        model = rf101(num_classes=args.data.NUM_CLASSES, pretrained=True)
    elif args.backbone == 'resnet50' :
        model = rf50(num_classes=args.data.NUM_CLASSES, pretrained=False) #TODO pretrained: True
    else : raise NameError('等待更新')
    model = model.cuda()
    if args.multi_gpu:
        # DistributedDataParallel
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # #TODO below is the checkpoints load from other's code:
    # saved_model = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    # model.module.load_state_dict(saved_model)
    #---------Below is temp----------
    # if args.checkpoint is not None:
    #     if os.path.isfile(args.checkpoint):
    #         load_checkpoint(model.module if args.multi_gpu else model, args.checkpoint)
    #     else:
    #         print('Provided checkpoint is not path to a file')
    #         return
    
   

    

    #Load optimizer

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
    
    optimizer = torch.optim.SGD(params=tencent_trick(model), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
    criterion = Segmentation_Loss()  #* 这里暂时尊重源代码，需要看论文到时候。
    criterion.cuda()

    # Set up metrics
    metrics = seg_metrics.StreamSegMetrics(args.data.NUM_CLASSES)
    if args.resume:
        model_state_file = args.resume
        model, _, _, start_epoch = load_checkpoint(args, model, optimizer, scheduler, model_state_file) #TODO 这里临时改了一下
        log.logger.info("Loaded checkpoint (starting from epoch {})".format(start_epoch))
    else:
        start_epoch = 0
 
    # Train Loop
    best_valid_iou = 0.0
    best_epoch = 0
    total_time = 0
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        if args.multi_gpu:
            train_dataloader.sampler.set_epoch(epoch)
            val_dataloader.sampler.set_epoch(epoch)
        train_avg_loss = train_loop(train_dataloader, model, criterion, optimizer, log, args, epoch,scaler)
        if epoch > 2 * args.epochs // 3:
            # torch.cuda.empty_cache()
            temp_iou = validation(val_dataloader, model, metrics)
            if args.local_rank == 0:
                if temp_iou > best_valid_iou:
                    best_valid_iou = temp_iou
                    best_epoch = epoch  
                    save_checkpoint(args, model, optimizer, scheduler, epoch,  "best_checkpoint.pth")
                log.logger.info("[Epoch %d/%d] valid mIoU %.4f" % (epoch + 1, args.epochs, temp_iou))
                log.logger.info("Best valid mIoU %.4f Epoch %d" % (best_valid_iou, best_epoch))
            # torch.cuda.empty_cache()
        else:
            temp_iou = 0.0
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time
        eta = base_utils.eta_compute(end_epoch_time, epoch, args.epochs)
        if args.local_rank == 0:
            log.logger.info("Epoch:%d, Train_Avegrage_loss: %.4f, Use Time:%d s, Eta: %.2f h", epoch + 1, train_avg_loss, end_epoch_time, eta)
            save_checkpoint(args, model, optimizer, scheduler, epoch,  "last_checkpoint.pth")
        scheduler.step()
def validation(val_loader, model, metrics):
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            images = sample["image"].cuda()
            targets = sample["label"].cuda()
            outputs = model(images) 
            outputs = nn.functional.interpolate(
            outputs, size=targets.size()[1:], mode="bilinear", align_corners=False)  
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = targets.cpu().numpy()
            metrics.update(targets, preds)
    score = metrics.get_results()
    Mean_Iou = score["Mean IoU"]
    return Mean_Iou
if __name__ == '__main__':
    main()  
     