import argparse
from utils.param_FLOPs_counter import model_info
from utils.Logger import Logger

from model.refinenet import rf101
parser = argparse.ArgumentParser(description='RefineNet Training With PyTorch')
parser.add_argument('--model_name', default='RefineNet', type=str,
                help='The model name')

parser.add_argument('--backbone', type=str, default='resnet50',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--backbone-path', type=str, default=None,
                    help='Path to chekcpointed backbone. It should match the'
                            ' backbone model declared with the --backbone argument.'
                            ' When it is not provided, pretrained model from torchvision'
                            ' will be downloaded.')

args = parser.parse_args()
net = rf101(num_classes=21, pretrained=True)

log = Logger('logs/'+ args.model_name+'_INFO.log',level='debug')
model_info(net,log,img_size=[512,512])
