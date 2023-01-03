
#from data.datasets import cityscapes, kd, coco, combine_dbs, sbd
#from data.segdatasets import Cityscapes, CityscapesPanoptic, COCOPanoptic
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from .datasets.pascal_voc import VOCSegmentation
def make_dataloader(args):
    root = args.data.DATASET_PATH 

    if args.data.NAME == 'VOC':
        train_dataset = VOCSegmentation(args, root, split='train')
        val_dataset = VOCSegmentation(args, root, split='val')

        if args.multi_gpu:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            train_shuffle = False
        else:
            train_sampler = None
            train_shuffle = True
            val_sampler = None
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=train_shuffle, sampler=train_sampler, collate_fn=None, pin_memory=True)    
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=val_sampler, collate_fn=None, pin_memory=True)    

        return train_dataloader,  val_dataloader
    elif args.data.NAME == 'cityscapes':
        if args.autodeeplab == 'train_seg':
            dataset_cfg = {
            'cityscapes': dict(
                root=args.data_path,
                split='train',
                is_train=True,
                crop_size=(args.image_height, args.image_width),
                mirror=True,
                min_scale=0.5,
                max_scale=2.0,
                scale_step_size=0.1,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )}
            train_set = Cityscapes(**dataset_cfg['cityscapes'])
            num_class = train_set.num_classes
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

            dataset_val_cfg = {
            'cityscapes': dict(
                root=args.data_path,
                split='val',
                is_train=False,
                crop_size=(args.eval_height, args.eval_width),
                mirror=True,
                min_scale=0.5,
                max_scale=2.0,
                scale_step_size=0.1,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )}
            val_set = Cityscapes(**dataset_val_cfg['cityscapes'])
            val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size//4), shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
        
        elif args.autodeeplab == 'train_seg_panoptic':
            dataset_cfg = {
            'cityscapes_panoptic': dict(
                root=args.data_path,
                split='train',
                is_train=True,
                crop_size=(args.image_height, args.image_width),
                mirror=True,
                min_scale=0.5,
                max_scale=2.0,
                scale_step_size=0.1,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                semantic_only=False,
                ignore_stuff_in_offset=True,
                small_instance_area=4096,
                small_instance_weight=3
            )}
            train_set = CityscapesPanoptic(**dataset_cfg['cityscapes_panoptic'])
            num_class = train_set.num_classes
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

            dataset_val_cfg = {
            'cityscapes_panoptic': dict(
                root=args.data_path,
                split='val',
                is_train=False,
                crop_size=(args.eval_height, args.eval_width),
                mirror=True,
                min_scale=0.5,
                max_scale=2.0,
                scale_step_size=0.1,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                semantic_only=False,
                ignore_stuff_in_offset=True,
                small_instance_area=4096,
                small_instance_weight=3
            )}
            val_set = Cityscapes(**dataset_val_cfg['cityscapes_panoptic'])
            val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size//4), shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
        else:
            raise Exception('autodeeplab param not set properly')

        return train_loader, val_loader, num_class


    elif args.data.NAME == 'coco':
        train_set = coco.COCOSegmentation(args, root, split='train')
        val_set = coco.COCOSegmentation(args, root, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
        test_loader = None
        return train_loader, train_loader, val_loader, test_loader, num_class

    elif args.data.NAME == 'kd':
        train_set = kd.CityscapesSegmentation(args, root, split='train')
        val_set = kd.CityscapesSegmentation(args, root, split='val')
        test_set = kd.CityscapesSegmentation(args, root, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader1 = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        train_loader2 = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

        return train_loader1, train_loader2, val_loader, test_loader, num_class

