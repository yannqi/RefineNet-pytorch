
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from ..utils import custom_transforms as tr

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

class VOCSegmentation(Dataset):
    def __init__(self, args, dataset_root, split):
        """crop_size: (h, w)"""
        self.split = split
        self.dataset_root = dataset_root
        self.args = args    
        #* Here Use the SDB Aug Dataset.
        self.id_file = '%s/ImageSets/Segmentation/%s' % (dataset_root, 'train_aug.txt' if self.split=='train' else 'val.txt')
        with open(self.id_file, 'r') as f:
            self.img_ids = f.read().split() # 拆分成一个个名字组成list
            
        
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.img_ids)))
        
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        self.img_id = self.img_ids[idx]
        _image, _target = self._read_voc_images()
        sample = {'image': _image, 'label': _target}
        if self.split == "train":
            return self._transform_train(sample)
        elif self.split == 'val':
            return self._transform_val(sample)



    def _read_voc_images(self):
        image= Image.open('%s/JPEGImages/%s.jpg' % (self.dataset_root, self.img_id)).convert("RGB")
        mask_image = Image.open('%s/SegmentationClassAugRaw/%s.png' % (self.dataset_root, self.img_id))
        # mask_image = Image.open('%s/SegmentationClass/%s.png' % (self.dataset_root, self.img_id)).convert("RGB")  #* Here Use the SDB Aug Dataset.
        return image, mask_image # PIL image 0-255

    def _transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def _transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size = self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
    
    def _label2colormap(self, target):
        """Convert label image to color image for visualization."""
        mask = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
        for i in range(len(VOC_COLORMAP)):
            mask[target == i] = VOC_COLORMAP[i]
        mask = Image.fromarray(mask)
        

def _voc_label_indices( mask_colormap ):
    """Convert the colormap image to label indices."""
    colormap2label = np.zeros(256**3, dtype=np.uint8) # torch.Size([16777216])
    for i, colormap in enumerate(VOC_COLORMAP):
        # 每个通道的进制是256，这样可以保证每个 rgb 对应一个下标 i
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    mask_colormap = np.array(mask_colormap.convert("RGB")).astype('int32')
    idx = ((mask_colormap[:, :, 0] * 256 + mask_colormap[:, :, 1]) * 256 + mask_colormap[:, :, 2]) 
    return colormap2label[idx] # colormap 映射 到colormaplabel中计算的下标

            
if __name__ == '__main__':
    #TODO Test the model.
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data.BASE_SIZE = 513
    args.data.CROP_SIZE = 513
    dataset_root = '/home/yangqi/dataspace/VOCdevkit/VOC2012' 
    voc_train = VOCSegmentation(args, dataset_root, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            segmap = sample['label'].numpy()
            segmap = segmap[jj]
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
