## Multiclass semantic segmentation on [cityscapes](https://www.cityscapes-dataset.com) and [kitti](http://www.cvlibs.net/datasets/kitti/eval_road.php) datasets.  

<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/UNET_2x_stuttgart_01.gif" height=320 /> </p>  

### Intro  

Semantic segmentation is no more than pixel-level classification and is well-known in the deep-learning community. There are several "state of the art" approaches for building such models. So basically we need a fully-convolutional network with some pretrained backbone for feature extraction to "map" input image with given masks (let's say, each output channel represents the individual class).  
Here is an example of cityscapes annotation:  

<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/download (72).png" height=300 /> </p>  

At this repo I want to show a way to train two most popular architectures - UNET and FPN (with pretty large `resnext50` encoders).  
Also, I want to give an idea of where we can use these semantic masks in the self-driving/robotics field: one of the **use cases** can be **generating "prior" for point cloud clustering** algorithms. But you can ask a question: why is semantic segmentation when in this case it's better to use panoptic/instance segmentation? Well, my answer will be: semantic segmentation models are a lot simpler and faster to understand and train.  

### Unet vs Feature Pyramid Network  

Both UNET and FPN uses features from the different scales and I'll quote really insightful words from the web about the difference between UNet and FPN:  
```  
...
 The main difference is that there is multiple prediction layers: one for each upsampling layer. 
 Like the U-Net, the FPN has laterals connection between the bottom-up pyramid (left) and the top-down pyramid (right).
 But, where U-net only copy the features and append them, FPN apply a 1x1 convolution layer before adding them. 
 This allows the bottom-up pyramid called “backbone” to be pretty much whatever you want.  
...
```  
Check out [the UNET paper](https://arxiv.org/pdf/1505.04597.pdf), which also gives the idea on separating instances (with borders predictions).  

<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/1_dKPBgCdJx6zj3MpED3lcNA.png" height=384 /> </p>

And [this presentation](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf) from the [FPN paper](https://arxiv.org/pdf/1901.02446.pdf) authors.  

<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/Screenshot from 2019-12-09 18-35-14.png" height=384 /> </p>

#### Models size (with default parameters and same encoders)  

To get model's short summary, I prefer using [torchsummary](https://github.com/sksq96/pytorch-summary).  
Torchsummary lib may require little hack to be able to work with FPN implementation.  
Make the folowing edits into the `torchsummary.py`:  
```
...
try:
    summary[m_key]["input_shape"] = list(input[0].size()) 
except:
    summary[m_key]["input_shape"] = list(input[0][0].size()) 
...
```  
Example (you will see the results in the stdout):  
```
python model_summary.py \
       --model_type fpn \
       --backbone resnext50 \
       --unet_res_blocks 0 \
       --input_size 3,512,1024
```  

`--model_type unet --backbone resnext50 --unet_res_blocks 1 --input_size 3,512,1024`:  
```
Total params: 92,014,472
Forward/backward pass size (MB): 10082.50 
Params size (MB): 351.01
```  

`--model_type unet --backbone resnext50 --unet_res_blocks 0 --input_size 3,512,1024`:  
```
Total params: 81,091,464 
Forward/backward pass size (MB): 9666.50
Params size (MB): 309.34
```  

`--model_type fpn --backbone resnext50 --input_size 3,512,1024`:  
```
Total params: 25,588,808
Forward/backward pass size (MB): 4574.11
Params size (MB): 97.61
```  

As we can see, FPN segmentation model is a lot "lighter" (so we can make larger batch size ;) ).  

#### Logs  
To monitor the training process, we can set up tensorboard (on CPU):  
```
CUDA_VISIBLE_DEVICES="" tensorboard --logdir /samsung_drive/semantic_segmentation/%MDOEL_DIR%/tensorboard  
```  
Logs are sending from the main training loop in `Trainer` class.  
Here are examples of typical training process (Unet for all cityscapes classes):  
<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/Screenshot from 2019-11-23 17-00-33.png" height=250 />  <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/Screenshot from 2019-11-23 17-00-41.png" height=250 />  </p>  

### Training configuration  

Core functionality is implemented in `Trainer` class (`./utils/trainer.py`).  
The training/evaluating pipeline is straight-forward: you just need to fill out the `train_config.yaml` according to the selected model and dataset.  
Let's take a look at the each module in training/evaluation configuration:  
 - `TARGET` - the dataset you want to train on - it affect the preprocessing stage (`cityscapes` or `kitti`);  
 - `PATH` - it's just the paths to train, validation and test datasets;  
 - `DATASET` - here we must set the image sizes, select classes to train and control augmentations;  
 - `MODEL` - declare model type, backbone and number of classes (affects the last layers of network);  
 - `TRAINING` - here are all training process properties: GPUs, loss, metric, class weights and etc.;  
 - `EVAL` - paths to store the predictions, test-time augmentation, flag, thresholds and etc.;  
There is an example of config file in the root dir.  
To use different configs, just pass them to the train/eval scripts as arguments:  
```
python train.py --config_path ./train_config.yaml
```  

#### Datasets  

For training, I've mostly used a fine-annotated part of the cityscapes dataset (~3k examples). It also exists a large amount of **coarse-annotated data**, and it is **obviously sufficient for pre-training**, but I didn't consider this part of the dataset in order to save time on training.  
After training on the cityscapes dataset (in case of road segmentation), you can easily use this model as initialization for the Kitti dataset to segment road/lanes.  
The cityscapes dataset also gives you a choice to use all classes or categories - as classes aggregated by certain properties.  
I've implemented all dataset-specific preprocessing in `cityscapes_utils.py` and `kitti_lane_utils.py` scripts.  

#### Loss and metric  

I used the [dice loss (which is equivalent to the F1 score)](https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/) as a default metric. In segmentation problems, it's usually applied intersection over union and dice metrics for evaluation. They're positively correlated, but the dice coefficient tends to measure some average performance for all classes and examples. Here is a nice visualization of **IoU (on the left)** and **dice (on the right)**:  

<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/random - New frame.jpg" height=200 /> </p>  

Now about the loss - I used weighted sum of binary cross-entropy and dice loss, as in binary classification [here](https://github.com/gasparian/PicsArtHack-binary-segmentation). You may also easily use IoU instead of dice since their implementation is very similar.  
BCE was calculated on logits for [numerical stability](https://towardsdatascience.com/sigmoid-activation-and-binary-crossentropy-a-less-than-perfect-match-b801e130e31).  

#### Augmentations and post-processing  

Augmentation is a well-established technique for dataset extension. What we do, is slightly modifying both the image and the mask. Here, I apply augmentations "on the fly" along with the batch generation, via the best-known library [albumentations](https://github.com/albumentations-team/albumentations).  
Usually, I end up with some mix of a couple spatial and RGB augmentations: like crops/flip + random contrast/brightness (you can check out it in `./utils/cityscapes_utils.py`).  
Also, sometimes you want to apply really hard augs, to imitate images from other "conditions distribution", like snow, shadows, rain and etc. Albumentations gives you that possibility via [this code](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library).  

Here is an original image:  
<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/download (73).png" height=320 /> </p>  

Here is a "darkened" flipped and slightly cropped image:  
<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/download (79).png" height=320 /> </p>  

This is a previous augmented image with random rain, light beam, and snow:  
<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/download (74).png" height=320 /> </p>  

<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/download (76).png" height=320 /> </p>  

<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/download (77).png" height=320 /> </p>  

Another way to use augmentations to increase model performance is to apply some "soft" **deterministic** affine transformations like flips and then average the results of the predictions (I've read the great analogy on how a human can look at the image from different angles and better understand what is shown there).  
This process called **test-time augmentation** or simply **TTA**. The bad thing is that we need to make predictions for each transform, which leads to larger inference time. Here is some visual explanation on how this works:  

<p align="center"> <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/temp_TTA.png" height=350 /> </p>  

*tsharpen is just (x_0^t + ... +x_i^t)/N*  

I use simple arithmetic mean, but you can try, for instance, geometric mean, tsharpen and etc. Check the code here: `/utils/TTA.py`.  
As the post-processing step, I detect and replace clusters of a certain area with background class, which leads to a "jitter" effect on a small and far situated masks (check out `/utils/utils.py-->DropClusters`).  

### Training results and weights  

Here I took the best of the 40 epochs of training on 2x down-sized images (512x1024) for 2 and 8 classes and 8x down-sized images for 20 classes (to fit the batch into GPU's memory).  
Models for 2-8 classes were trained in two stages: on smaller images at first - 256x512 and then only 2x resized - 512x1024.  

Dice metric comparison table:  

Classes #                     | Unet   | FPN   | Size
:----------------------------:|:------:|:-----:|:----:
2 (road segmentation) | 0.956 ([weights](https://drive.google.com/open?id=1L7mYrM0oBFDvO1OxU7NOZYrNWWNUhRkR))  | 0.956 ([weights](https://drive.google.com/open?id=10XamX7t5T59evY_1OiJEJQBSAKDn8tw8)) | 256x512 >> 512x1024
8 (categories only)   | 0.929 ([weights](https://drive.google.com/open?id=1DX5Akcu5vRGnkAcH2tljtMSPkbnWH6ZV))  | 0.931 ([weights](https://drive.google.com/open?id=13TxEjLemfjMvqBEyfmQJ3pbOgK32fQxX)) | 256x512 >> 512x1024
20                    | 0.852 ([weights](https://drive.google.com/open?id=13NZA-zajFbMGqOsMK-1ldIipbQMHz4vo))  | 0.858 ([weights](https://drive.google.com/open?id=1_xqp5h8eUtOnv_EIQbFi3BMNG0pLLiXX)) | 128x256  

8 classes:  

Model  | void       | flat   | construction | object     | nature | sky   | human     | vehicle
:-----:|:----------:|:------:|:------------:|:----------:|:------:|:-----:|:---------:|:-------:
FPN    | **0.769**  | 0.954  | 0.889        | **0.573**  | 0.885  | 0.804 | **0.492** | 0.897  
UNET   | **0.750**  | 0.958  | 0.888        | **0.561**  | 0.884  | 0.806 | **0.479** | 0.890  

20 classes:  

Model  | road   | sidewalk   | building     | wall      | fence      | pole      | traffic light | traffic sign | vegetation | terrain   | sky   | person | rider      | car   | truck     | bus       | train     | motorcycle | bicycle | unlabeled  
:-----:|:------:|:----------:|:------------:|:---------:|:----------:|:---------:|:-------------:|:------------:|:----------:|:---------:|:-----:|:------:|:----------:|:-----:|:---------:|:---------:|:---------:|:----------:|:-------:|:---------:
FPN    | 0.943  | 0.562      | 0.777        | 0.011     | **0.046**  | 0.041     | 0.318         | 0.128        | 0.808      | 0.178     | 0.747 | 0.132  | **0.0132** | 0.759 | **0.010** | **0.022** | **0.013** | **0.005**  | 0.072   | **0.216**  
UNET   | 0.944  | 0.608      | 0.785        | **0.020** | 0.017      | **0.131** | 0.321         | **0.161**    | 0.822      | **0.236** | 0.765 | 0.141  | 0.000      | 0.780 | 0.001     | 0.002     | 0.001     | 0.000      | 0.056   | 0.112  

<img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/FPN8.png" height=280 >  <img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/FPN20.png" height=280 >  

So what is interesting, that I expected to see better performance on multiclass problems by FPN architecture, but the thing is on average both UNET and FPN gives pretty close dice metric.  
Yes, there are a couple of classes that the FPN segmentation model detects better (marked in the table), but the absolute dice metric values of such classes, are not so high.  

***Summary***:  
*In general, if you're dealing with some generic segmentation problem with pretty large, nicely separable objects - it seems that the **FPN could be a good choice for both binary and multiclass segmentation** in terms of segmentation quality and computational effectiveness, **but** at the same time I've noticed that **FPN gives more small gapes in masks** opposite to the UNET. Check out videos below:*  

Prediction on cityscapes demo videos (Stuttgart):  

Classes # | UNET | FPN 
:--------:|:-----:|:-----
2         | [00](https://youtu.be/dHnidGY_Lwc), [01](https://youtu.be/RURCE3K7OeA), [02](https://youtu.be/OrAe5DiYWQk) | [00](https://youtu.be/RUu6upRSi20), [01](https://youtu.be/innUjjzpQ8s), [02](https://youtu.be/cGZTEw16rQg)  
8         | [00](https://youtu.be/hmIV17M7Gf8), [01](https://youtu.be/lW43CHLNL5k), [02](https://youtu.be/a2HjDz_IMMg) | [00](https://youtu.be/7qGSZ9XypkE), [01](https://youtu.be/6PhdoajzwNQ), [02](https://youtu.be/O0_Jzrfmgqk)  
20        | [00](https://youtu.be/EpN4Jx60pXI), [01](https://youtu.be/X1Oa2x5BAkg), [02](https://youtu.be/rkm6OpPCZY0) | [00](https://youtu.be/DzyLExn0M54), [01](https://youtu.be/OJyR_4U7PV8), [02](https://youtu.be/Wez8wFR3QOY)  


I used `ffmpeg` for making videos from images sequence on Linux:  
```
ffmpeg -f image2 -framerate 20 \
       -pattern_type glob -i 'stuttgart_00_*.png' \
       -c:v libx264 -pix_fmt yuv420p ../stuttgart_00.mp4
```  

### Reference  

Check out [this awesome](https://github.com/qubvel/segmentation_models.pytorch) repo with high-quality implementations of the basic semantic segmentation algorithms.  

### Reproducibility  

Again, I strongly suggest to use [Deepo](https://github.com/ufoym/deepo) as a simple experimental enviroment.  
When you've done with your code - better build your own docker container and keep the last version on somewhere like [dockerhub](https://hub.docker.com/).  
Anyway, here are some key dependencies for these repo:  
```
pip install --upgrade tqdm \
                      torchsummary \
                      tensorboardX \
                      albumentations==0.4.1 \
                      torch==1.1.0 \ 
                      torchvision==0.4.0
```  
