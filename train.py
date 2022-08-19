import os
import argparse
import yaml
import time

import numpy as np
import torch 
import torch.nn as nn 
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from model.refinenet import rf101
from utils.Logger import Logger
from utils.multi_gpu import init_distributed_mode
from data.coco import COCOSegmentation
from model.utils.loss import Segmentation_Loss
from model.utils.train_utils import train_loop, tencent_trick, load_checkpoint



def main():
    parser = argparse.ArgumentParser(description='RefineNet Training With PyTorch')
    parser.add_argument('--model_name', default='RefineNet', type=str,
                        help='The model name')
    parser.add_argument('--model_config', default='configs/refinenet.yaml', 
                        metavar='FILE', help='path to model cfg file', type=str,)
    parser.add_argument('--data_config', default='configs/coco_21class.yaml', 
                        metavar='FILE', help='path to data cfg file', type=str,)
    parser.add_argument('--device_gpu', default='0,1,2', type=str,
                        help='Cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--save_step', default=20, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--checkpoint', default='checkpoints/num_classes21/refinenet101_voc.pth', help='The checkpoint path')


    parser.add_argument('--save', type=str, default='checkpoints',
                        help='save model checkpoints in the specified directory')
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
    parser.add_argument('--num_workers', type=int, default=8) 
    
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--backbone-path', type=str, default=None,
                        help='Path to chekcpointed backbone. It should match the'
                             ' backbone model declared with the --backbone argument.'
                             ' When it is not provided, pretrained model from torchvision'
                             ' will be downloaded.')
    parser.add_argument('--report-period', type=int, default=800, help='Report the loss every X times.')
    
    parser.add_argument('--save-period', type=int, default=5, help='Save checkpoint every x epochs (disabled if < 1)')
    
    # Multi Gpu
    parser.add_argument('--multi_gpu', default=False, type=bool,
                        help='Whether to use multi gpu to train the model, if use multi gpu, please use by sh.')
    
    #others 
    parser.add_argument('--amp', action='store_true', default = False,
                        help='Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')

    args = parser.parse_args()
    
    
    #Config Load
    
    data_cfg_path = open(args.data_config)
    # 引入EasyDict 可以让你像访问属性一样访问dict里的变量。
    from easydict import EasyDict as edict
    data_cfg = yaml.full_load(data_cfg_path)
    data_cfg = edict(data_cfg) 
    args.data = data_cfg
    
    cfg_path = open(args.model_config)
    cfg = yaml.full_load(cfg_path)
    cfg = edict(cfg) 
    args.model = cfg
    #Random seed
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    
    # Initialize Multi GPU 

    if args.multi_gpu == True :
        init_distributed_mode(args)
    else: 
        # Use Single Gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} device')
        args.device = device   
        args.NUM_gpu = 1
        args.local_rank = 0
        
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True

 
    #Logger
    log_path = '{}-{}-lr-{}-{}'.format(args.model_name, data_cfg.NUM_CLASSES, args.lr, time.strftime('%Y%m%d-%H'))
    
    log = Logger('logs/'+log_path+'.log',level='debug')
    
    #Initial Logging
    if args.local_rank == 0:
        log.logger.info('gpu device = %s' % args.device_gpu)
        log.logger.info('args = %s', args)
        log.logger.info('data_cfgs = %s', data_cfg)

    #Pre Data

    train_dataset = COCOSegmentation(args.data.DATASET_PATH, args, split='train')
    
    
    if args.multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       num_workers=args.num_workers ,shuffle=train_shuffle ,sampler=train_sampler , collate_fn=None,pin_memory=True)


    #Load Model 
    if args.backbone == 'resnet101' :
        net = rf101(num_classes=args.data.NUM_CLASSES, pretrained=True)
            
    else : raise NameError('等待更新')
    net = net.cuda()
    if args.multi_gpu:
        # DistributedDataParallel
        net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)

    #TODO temp:
    saved_model = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    net.module.load_state_dict(saved_model)
    #---------Below is temp----------
    
    # if args.checkpoint is not None:
    #     if os.path.isfile(args.checkpoint):
            
    #         load_checkpoint(net.module if args.multi_gpu else net, args.checkpoint)
    #     else:
    #         print('Provided checkpoint is not path to a file')
    #         return
    

    #Load optimizer

    #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
    
    optimizer = torch.optim.SGD(params=tencent_trick(net), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
    criterion = Segmentation_Loss()
    

    criterion.cuda()






    #  train
     
    total_time = 0
    if args.local_rank == 0:
        log.logger.info("'Train on {} samples".format(train_dataset.__len__()))
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)  # Automatic Mixed Precision
    for epoch in range(args.epochs):
        if args.multi_gpu :
            train_dataloader.sampler.set_epoch(epoch)
        start_epoch_time = time.time()
        train_loop(net,criterion,scaler,epoch,optimizer,train_dataloader,args,log)
        scheduler.step()
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            log.logger.info('Epoch:',epoch,'Use Time:', end_epoch_time) 
        
        if epoch % args.save_period == 0 and args.local_rank == 0 :
            print("saving model...")
            obj = {}
            if args.multi_gpu:
                obj['model'] = net.module.state_dict()
            else:
                obj['model'] = net.state_dict()
            save_path = os.path.join(args.save,  f'Class_{args.data.NUM_CLASSES}_epoch_{epoch}.pt')
            torch.save(obj, save_path)
            log.logger.info('model path:', save_path)
    if args.local_rank == 0:
        log.logger.info('total time:', total_time )

   
if __name__ == '__main__':
    main()  
     