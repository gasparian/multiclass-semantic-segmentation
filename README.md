## Multiclass semantic segmentation on [cityscapes](https://www.cityscapes-dataset.com) and [kitti](http://www.cvlibs.net/datasets/kitti/eval_road.php) datasets.  

<img src="https://github.com/gasparian/semantic_segmentation_experiments/blob/master/imgs/UNET_2x_stuttgart_01.gif" width="700" height="400" />  

### Dependencies:  
Again, I strongly suggest to use [Deepo](https://github.com/ufoym/deepo) as simple experimental enviroment.  
When you've got your "final" version of code - better build your own docker container and keep the last version on somewhere like [dockerhub](https://hub.docker.com/).  
Anyway, here are some key dependencies:  
```
pip install --upgrade tqdm \
                      torchsummary \
                      albumentations==0.4.1 \
                      torch==1.1.0 \ 
                      torchvision==0.4.0
```  

### Problem statement  
The semantic segmentation problem itself well-known in the deep-learning community, and there are already several "state of the art" approaches to build such models. So basically we need fully-convolutional network with some pretrained backbone for feature extraction, to "map" input image with given masks (let's say each output channel represents the individual class).  
Here are some examples of semantic segmentation from cityscapes training set:  

Original | Mask  
:-------------------------:|:-------------------------:
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_3_orig_mask.png"> | <img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_3_edited_mask.png">  

In this repo I wanted to show a way to train two most popular architectures - UNET and FPN (with pretty large resnext50 encoder).  
Before we go into more details, I want to give an idea of where we can use these semantic masks in self-driving/robotics field: one of the use cases can be generating "prior" for pointclouds clustering algorihms. But you can ask a quation: why is semantic segmentation, when at this case it's better to use panoptic segmentation? Well, my answer will be: semantic segmentation models is a lot more simplier to understand and train, including the computational resources consumption ;)  

### Unet vs Feature Pyramid Network  

Both UNET and FPN uses the same conception - to use features from the different scales and I'll use really insightful words from the web about the difference between Unet and FPN:  
```
 The main difference is that there is multiple prediction layers: one for each upsampling layer. Like the U-Net, the FPN has laterals connection between the bottom-up pyramid (left) and the top-down pyramid (right). But, where U-net only copy the features and append them, FPN apply a 1x1 convolution laye45r before adding them. This allows the bottom-up pyramid called “backbone” to be pretty much whatever you want.  
```  
Check out [this UNET paper](https://arxiv.org/pdf/1505.04597.pdf), which also give the idea of separating instances.  
<img src="https://github.com/gasparian/PicsArt-Hack-binary_segmentation/blob/master/pics/ex_3_orig_mask.png" height=384>  

And this [presentation](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf) from the [FPN paper](https://arxiv.org/pdf/1901.02446.pdf) authors.  

#### Models size (with default parameters and same encoders)  

To get model's short summary, I prefer using [torchsummary](https://github.com/sksq96/pytorch-summary).  
Torchsummary lib may require little hack to be able to work with FPN inplementation.  
Make the folowing edits into the `torchsummary.py`:  
```
try:
    summary[m_key]["input_shape"] = list(input[0].size()) 
except:
    summary[m_key]["input_shape"] = list(input[0][0].size()) 
```  
Anf run the script (you will see the results in the stdout):  
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

As we can see, FPN segmentation model is a lot "lighter" (so faster to train).  

#### Logs  
The last step before training, we can set up tensorboard (on CPU):  
```
CUDA_VISIBLE_DEVICES="" tensorboard --logdir /samsung_drive/semantic_segmentation/%MDOEL_DIR%/tensorboard  
```  

### Training and results  


### Convert images to video:  

```
ffmpeg -f image2 -framerate 20 \
       -pattern_type glob -i 'stuttgart_00_*.png' \
       -c:v libx264 -pix_fmt yuv420p ../stuttgart_00.mp4
```  

https://youtu.be/hmIV17M7Gf8 - UNET 8 classes 00;  
https://youtu.be/lW43CHLNL5k - UNET 8 classes 01;  
https://youtu.be/a2HjDz_IMMg - UNET 8 classes 02;  

https://youtu.be/7qGSZ9XypkE - FPN 8 classes 00;  
https://youtu.be/6PhdoajzwNQ - FPN 8 classes 01;  
https://youtu.be/O0_Jzrfmgqk - FPN 8 classes 02;  

https://youtu.be/EpN4Jx60pXI - UNET 20 classes 00;  
https://youtu.be/X1Oa2x5BAkg - UNET 20 classes 01;  
https://youtu.be/rkm6OpPCZY0 - UNET 20 classes 02;  

https://youtu.be/DzyLExn0M54 - FPN 20 classes 00;  
https://youtu.be/OJyR_4U7PV8 - FPN 20 classes 01;  
https://youtu.be/Wez8wFR3QOY - FPN 20 classes 02;  

```
ffmpeg -f image2 -framerate 20 \
       -pattern_type glob -i 'stuttgart_00_*.png' \
       -c:v libx264 -pix_fmt yuv420p ../stuttgart_00.mp4
```  

#### Links:  

On logits vs activations: https://towardsdatascience.com/sigmoid-activation-and-binary-crossentropy-a-less-than-perfect-match-b801e130e31;  

