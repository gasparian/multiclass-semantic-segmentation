
#### Instal dependencies:  
```
pip install --upgrade tqdm \
                      albumentations==0.4.1 \
                      torch==1.1.0 \ 
                      torchvision==0.4.0
```  

#### Unet vs FPN:  

http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf  

About difference between UNet and FPN:  
```
 The main difference is that there is multiple prediction layers: one for each upsampling layer. Like the U-Net, the FPN has laterals connection between the bottom-up pyramid (left) and the top-down pyramid (right). But, where U-net only copy the features and append them, FPN apply a 1x1 convolution laye45r before adding them. This allows the bottom-up pyramid called “backbone” to be pretty much whatever you want.  
```  

Torchsummary lib may require little hack to be able to work with FPN inplementation.  
Make the folowing edits into the `torchsummary.py`:  
```
try:
    summary[m_key]["input_shape"] = list(input[0].size()) 
except:
    summary[m_key]["input_shape"] = list(input[0][0].size()) 
```  

You will see the results in the stdout:  
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

#### Experiments plan:  
1. Prepare images and labels (ohe - seaprated channels for each class);  
        https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py  
2. Select only categories-classes;  
3. Apply size progression while training;  
4. Try UNet with and without Res. blocks in decoder (ResNext50/101 backbone);  
5. Try FPN (ResNext50/101 backbone);  
6. Try Snow/Rain augmentations (hard_augs) on the best performed model;  
7. Try TTA on the best one;  
8. Make demo videos with segmentation;  
9. Make small student network (mobilenet encoder) and train knowledge distillation with coarse and fine datasets:  
        https://arxiv.org/pdf/1903.04197.pdf
        https://github.com/irfanICMLL/structure_knowledge_distillation

To do:   
 - get validation statistics for each class;  
 - add distillation to train smaller network (??);  
 - fuse masks with lidar data for kitti dataset;  

In progress:  
- ;  

Done:  
 - calculate number of parameters of used networks; 
 - retrain kitti on 320x1024 instead of 256x1024;  
 - retrain UNETs for roads with 1 class and save the results on board;  
 - add the KITTI if-else statement in train and eval scripts;  
 - train on KITTI dataset;  

#### Logs:  
Run tensorboard on CPU:  
```
CUDA_VISIBLE_DEVICES="" tensorboard --logdir /samsung_drive/semantic_segmentation/%MDOEL_DIR%/tensorboard  
```  

#### Convert images to video:  

```
ffmpeg -f image2 -framerate 20 \
       -pattern_type glob -i 'stuttgart_00_*.png' \
       -c:v libx264 -pix_fmt yuv420p ../stuttgart_00.mp4
```  

#### Links: 

https://miro.com/app/board/o9J_kwbzsfE=/  

On logits vs activations: https://towardsdatascience.com/sigmoid-activation-and-binary-crossentropy-a-less-than-perfect-match-b801e130e31;  

