
Instal dependencies:  
```
pip install --upgrade tqdm \
                      albumentations==0.4.1 \
                      torch==1.1.0 \ 
                      torchvision==0.4.0
```  

About difference between UNet and FPN:  
 The main difference is that there is multiple prediction layers: one for each upsampling layer. Like the U-Net, the FPN has laterals connection between the bottom-up pyramid (left) and the top-down pyramid (right). But, where U-net only copy the features and append them, FPN apply a 1x1 convolution layer before adding them. This allows the bottom-up pyramid called “backbone” to be pretty much whatever you want.  

To do:  
 - add tensorboard logs;  
 - add current learning rate to the tensorboard;  
 - implenent train script:  
 - add predictions script;  
 - add distillation to train smaller network (??);  
