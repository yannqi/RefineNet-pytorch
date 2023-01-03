import argparse
import os
import time

import numpy as np
import torch
import yaml

from data.datasets.coco import COCOSegmentation

from model.evaluate import evaluate
from model.refinenet import rf101
#from utils.model_load_save import load_checkpoint
from utils.Logger import Logger
import utils.base_utils as base_utils

def load_checkpoint(model, checkpoint):
    """Load model from checkpoint."""
    
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint, map_location=torch.device('cpu'))
    
    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)

# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
def main():
    parser = argparse.ArgumentParser(description='RefineNet Training With PyTorch')
    parser.add_argument('--model_name', default='RefineNet', type=str,
                        help='The model name')
    parser.add_argument('--model_config', default='configs/refinenet.yaml', 
                        metavar='FILE', help='path to model cfg file', type=str,)
    parser.add_argument('--data_config', default='configs/coco_21class.yaml', 
                        metavar='FILE', help='path to data cfg file', type=str,)
    parser.add_argument('--device_gpu', default='4', type=str,
                        help='Cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--checkpoint', default="checkpoints/Class_21_epoch_90.pt", help='The checkpoint path')

    # Hyperparameters
    parser.add_argument('--batch_size', '--bs', type=int, default=10,
                        help='number of examples for each iteration')
    parser.add_argument('--num_workers', type=int, default=8) 
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    
    # Multi Gpu
    parser.add_argument('--multi_gpu', default=False, type=bool,
                        help='Whether to use multi gpu to train the model, if use multi gpu, please use by sh.')
    
    #others 
    parser.add_argument('--amp', action='store_true', default = False,
                        help='Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')
    args = parser.parse_args()
    
    
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

    
    # Initialize Multi GPU 

    if args.multi_gpu == True :
        base_utils.init_distributed_mode(args)
    else: 
        # Use Single Gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} device')
        args.device = device   
        args.NUM_gpu = 1
        args.local_rank = 0

    
    #The learning rate is automatically scaled 
    # (in other words, multiplied by the number of GPUs and multiplied by the batch size divided by 32).
    #Logger
    log_path = 'Test-{}-lr-{}-{}'.format(args.model_name, data_cfg.NAME, time.strftime('%Y%m%d-%H'))
    
    log = Logger('logs/'+log_path+'.log',level='debug')

    #Initial Logging
    if args.local_rank == 0:
        log.logger.info('gpu device = %s' % args.device_gpu)
        log.logger.info('args = %s', args)
        log.logger.info('data_cfgs = %s', data_cfg)
    
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True

    
    # Pre dataset       
    val_dataset = COCOSegmentation(args.data.DATASET_PATH, args, split='val')
 
    
    
    if args.multi_gpu:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)     
    else:

        val_sampler = None


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,  # Note: distributed sampler is shuffled :(
                                sampler=val_sampler,
                                num_workers=args.num_workers)
    
    
    # Load model
    if args.backbone == 'resnet101' :
        net = rf101(num_classes=args.data.NUM_CLASSES, pretrained=True)
    else : raise NameError('等待更新')
    net = net.cuda()
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(net.module if args.multi_gpu else net, args.checkpoint)
            #model, _, _, start_epoch = load_checkpoint(args, net, None, None, args.checkpoint)
        else:
            print('Provided checkpoint is not path to a file')
            return
    
    evaluate(net, val_loader, log, args)



if __name__ == '__main__':
    main()