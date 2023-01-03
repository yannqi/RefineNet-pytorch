#exmaple: 1 node,  2 GPUs per node (2GPUs)

CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22227 \
    train.py --multi_gpu=True \
    --data_config='configs/VOC.yaml' \
    --device_gpu='0,1' \
    --lr=0.001 \
    --base_size=513 \
    --crop_size=513 \
    --batch_size=24 \
    --report_period=600 \
    --epochs=300 \
    --backbone='resnet50' \
    --amp \
    --resume="/data/yangqi/checkpoints_save/refine_net/checkpoints/RefineNet/VOC-21-20221029-22/best_checkpoint.pth" \
#The following is for multi-computer situations
# CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=localhost \
#     --master_port=22222 \
#     train.py





# https://zhuanlan.zhihu.com/p/360405558
# https://juejin.cn/post/7044336367588868109