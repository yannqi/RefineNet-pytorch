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

from model.refinenet import rf101
from utils.Logger import Logger
from utils.multi_gpu import init_distributed_mode
from data.coco import COCOSegmentation
from model.utils.loss import Segmentation_Loss




def main():
    parser = argparse.ArgumentParser(description='RefineNet Training With PyTorch')
    parser.add_argument('--model_name', default='RefineNet', type=str,
                        help='The model name')
    parser.add_argument('--model_config', default='configs/refinenet.yaml', 
                        metavar='FILE', help='path to model cfg file', type=str,)
    parser.add_argument('--data_config', default='configs/coco.yaml', 
                        metavar='FILE', help='path to data cfg file', type=str,)
    parser.add_argument('--device_gpu', default='3,4', type=str,
                        help='Cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--checkpoint', default='checkpoints/refinenet101_voc.pth', help='The checkpoint path')

    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    




    parser.add_argument('--save', type=str, default='checkpoints',
                        help='save model checkpoints in the specified directory')
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'evaluation', 'benchmark-training', 'benchmark-inference'])
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='number of epochs for training') #default 65
    parser.add_argument('--evaluation', nargs='*', type=int, default=[21, 31, 37, 42, 48, 53, 59, 64],
                        help='epochs at which to evaluate')

    parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54],
                        help='epochs at which to decay learning rate')
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--seed', '-s', default = 42 , type=int, help='manually set random seed for torch')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default= 1e-5  ,
                        help='learning rate for SGD optimizer')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum argument for SGD optimizer')
    parser.add_argument('--weight_decay', '--wd', type=float, default=0.0005,
                        help='weight-decay for SGD optimizer')
    parser.add_argument('--batch_size', '--bs', type=int, default=4,
                        help='number of examples for each iteration')
    parser.add_argument('--num_workers', type=int, default=8) 
    
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--backbone-path', type=str, default=None,
                        help='Path to chekcpointed backbone. It should match the'
                             ' backbone model declared with the --backbone argument.'
                             ' When it is not provided, pretrained model from torchvision'
                             ' will be downloaded.')
    parser.add_argument('--report-period', type=int, default=100, help='Report the loss every X times.')
    # TODO add by yourself.
    # parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    
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
    log_path = '{}-{}-lr-{}-{}'.format(args.model_name, data_cfg.NAME, args.lr, time.strftime('%Y%m%d-%H'))
    
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
        #val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True
    
    train_dataLoader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       num_workers=args.num_workers ,shuffle=train_shuffle ,collate_fn=None,pin_memory=True)

    # val_data = dataset.NYUDV2Dataset(cfg.images, cfg.labels, cfg.depths, cfg.test_split)
    # val_dataLoader = data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True)
    #-------load net-------
    

    net = rf101(num_classes=args.data.NUM_CLASSES, pretrained=True)
    if args.checkpoint :
        net.load_state_dict(torch.load(args.checkpoint))  
    

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
    criterion = Segmentation_Loss()
    
    net = net.cuda()
    criterion.cuda()

    if args.multi_gpu:
        # DistributedDataParallel
        net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)

# # train
    if args.local_rank == 0:
        log.logger.info("'Train on {} samples".format(train_dataset.__len__()))
    
    for epoch in range(args.epochs):
        if args.local_rank == 0:
            log.logger.info('Epoch {0}/{1}\n'.format(epoch + 1, args.epochs))
        total_loss = 0
        step_start = time.time()
        for i, sample in enumerate(train_dataLoader):
            
            image = sample["image"].cuda()
            target = sample['label'].cuda()
            image_var = torch.autograd.Variable(image).float()
            target_var = torch.autograd.Variable(target).long()

            # target = torch.squeeze(target,1)
            # target = target.long()

            optimizer.zero_grad()
            prediction = net(image_var)
            prediction = nn.functional.interpolate(
            prediction, size=target_var.size()[1:], mode="bilinear", align_corners=False)
            loss = criterion(prediction,target_var)
            loss.backward()
            total_loss += loss
            optimizer.step()
            scheduler.step(loss) 
            if i % 2 == 0 :  
                step_stop = time.time()
                use_time = int(step_stop - step_start)
                mean_loss =  total_loss/i
                step_start = time.time()
                
                if args.local_rank == 0:
                    log.logger.info('-TRAINED: {0:10d}/{1} \n - USED: {2} s \n- loss: {3:.4f}  - mean loss: {4:.4f}'
                    .format((i + 1) * args.batch_size,
                            train_dataset.__len__(), use_time,loss,mean_loss))
    if args.local_rank == 0:               
        torch.save(net.state_dict(), 'outputs/COCO_refinenet/parameter_epoch'+str(epoch)+'.pkl', _use_new_zipfile_serialization=True)






if __name__ == '__main__':
    main()  
     