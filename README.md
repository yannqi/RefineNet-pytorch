# RefineNet For PyTorch (COCO Dataset)

This repository provides a script and recipe to train the RefineNet model on the COCO Dateset.
The method belongs to paper [RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/pdf/1611.06612.pdf) .

## Feature support matrix

Copy from [NVIDIA SSD pytorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)

The following features are supported by this model.

| **Feature** | **RefineNet  PyTorch** |
|:---------:|:----------:|
|[AMP](https://pytorch.org/docs/stable/amp.html)                                        |  Yes |
|[DDP Multi GPU](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)               |  Yes |
|[NVIDIA DALI](https://docs.nvidia.com/deeplearning/sdk/dali-release-notes/index.html)  |  No |

#### Features

[AMP](https://pytorch.org/docs/stable/amp.html) is an abbreviation used for automatic mixed precision training.

[DDP](https://nvidia.github.io/apex/parallel.html) stands for DistributedDataParallel and is used for multi-GPU training.

[NVIDIA DALI](https://docs.nvidia.com/deeplearning/sdk/dali-release-notes/index.html) - DALI is a library accelerating data preparation pipeline.
To accelerate your input pipeline, you only need to define your data loader
with the DALI library. For details, see example sources in this repo or see
the [DALI documentation](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)



## Usage

1. Clone the repository.
```
git clone https://gitee.com/yann_qi/RefineNet-pytorch.git

```

or

```
git clone https://github.com/yannqi/RefineNet-pytorch.git

```
2. Download and preprocess the dataset.

    The RefineNet model was trained on the COCO 2017 dataset. You can download the dataset on  [COCO Download](http://cocodataset.org/#download).

    **NOTE:** Make the dataset root like below:

        └──  COCO 
            ├──images
                ├── train2017: All train images(118287 images)
                ├── val2017: All validate images(5000 images)
            ├── annotations
                ├── instances_train2017.json
                ├── instances_val2017.json
                ├── captions_train2017.json
                ├── captions_val2017.json
                ├── person_keypoints_train2017.json
                └── person_keypoints_val2017.json
            └── coco_labels.txt

3. Config Setting 

    Set the config in the `data/coco.yaml`

4. Train the model.(Unnecessary, you can download the pretrained checkpoint.)


   - Single GPU
   
    `sh scripts/single_gpu.sh`
   - Multi GPU

    `sh scripts/multi_gpu.sh`

5.  Evaluate the model on the COCO dataset.

    Just run the `python test.py`


### Checkpoint Download

Wait to update!

### Results
Wait to update!
<!-- ```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.250
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.255
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.074
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.268
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.237
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.550
```

```
Model FLOPs: 20.213G Params: 22.895M
Model summary: 173 layers, 22894902 parameters, 22894902 gradients

``` -->




### Data preprocessing
Wait to update!
<!-- Before we feed data to the model, both during training and inference, we perform:
* JPEG decoding
* normalization with a mean =` [0.485, 0.456, 0.406]` and std dev = `[0.229, 0.224, 0.225]`
* encoding bounding boxes
* resizing to 512x512

Additionally, during training, data is:
* randomly shuffled
* samples without annotations are skipped -->

#### Data augmentation

Wait to update!






