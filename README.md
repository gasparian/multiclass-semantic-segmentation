
Instal dependencies:  
```
pip install --upgrade tqdm \
                      albumentations==0.4.1 \
                      torch==1.1.0 \ 
                      torchvision==0.4.0
```  

About difference between UNet and FPN:  
 The main difference is that there is multiple prediction layers: one for each upsampling layer. Like the U-Net, the FPN has laterals connection between the bottom-up pyramid (left) and the top-down pyramid (right). But, where U-net only copy the features and append them, FPN apply a 1x1 convolution laye45r before adding them. This allows the bottom-up pyramid called “backbone” to be pretty much whatever you want.  

Experiments plan:  
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

In progress:  
 - fuse masks with lidar data for kitti dataset;  
 - calculate number of parameters of used networks; 

Done:  
 - retrain UNETs for roads with 1 class and save the results on board;  
 - add the KITTI if-else statement in train and eval scripts;  
 - train on KITTI dataset;  


#### Logs:  
Run tensorboard on CPU:  
```
CUDA_VISIBLE_DEVICES="" tensorboard --logdir /samsung_drive/semantic_segmentation/%MDOEL_DIR%/tensorboard  
```  

#### Links: 

https://miro.com/app/board/o9J_kwbzsfE=/  

On logits vs activations: https://towardsdatascience.com/sigmoid-activation-and-binary-crossentropy-a-less-than-perfect-match-b801e130e31;  

