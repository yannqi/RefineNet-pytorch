from logging import error
import torch
import os
def save_checkpoint(args, model, optimizer, scheduler, epoch, checkpoint_name, model_ema=None):
    """Save the checkpoint.
    Args:
        epoch (int): current epoch.
        checkpoint_name (str): checkpoint name.
    """
    if model_ema is not None:
        if hasattr(model_ema.ema, 'module'):
            model_ema.ema = model_ema.ema.module
        torch.save({
            'start_epoch': epoch + 1,
            'state_dict': model_ema.ema.state_dict(),
            # 'current_lr': args.lr,
        }, os.path.join(args.save, checkpoint_name))
    else:
        if hasattr(model, 'module'):
            model = model.module
        torch.save({
                'start_epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                #'criterion': criterion.state_dict(),
                # 'current_lr': args.lr,
            }, os.path.join(args.save, checkpoint_name))
    
    
def load_checkpoint(args, model, optimizer, scheduler, model_path):
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))   
        start_epoch = checkpoint['start_epoch'] 
        model.module.load_state_dict(checkpoint['state_dict'], strict= False) if args.multi_gpu else model.load_state_dict(checkpoint['state_dict'], strict= False)
        if optimizer is None:
            pass
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if optimizer is None:
            pass 
        else:
            scheduler.load_state_dict(checkpoint['scheduler'])
        #current_lr = checkpoint['current_lr'] 
        return model, optimizer, scheduler, start_epoch
    else: raise error("The checkpoint path is wrong!")

def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters share common prefix 'module.'
    '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(missing_keys))
    print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))

    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        # new_dict = {}
        # for k in pretrained_dict.keys():
        #     if "heads" in k:
        #         continue
        #     else:
        #         new_dict[k] = pretrained_dict[k]
        # pretrained_dict = new_dict
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model