from torch.utils.data import Dataset
from albumentations import HorizontalFlip, RandomCrop, Resize, \
                           RandomBrightness, RandomContrast, RandomGamma, \
                           RandomFog, RandomRain, RandomShadow, RandomSnow, RandomSunFlare, \
                           Normalize, Compose, OneOf

class CustomDataset(Dataset):
    
    def __init__(self, hard_augs=False, resize=None, orig_size=None):
        self.orig_size = orig_size
        self.orig_h, self.orig_w = self.orig_size 
        self.h, self.w = self.orig_h, self.orig_w
        self.resize = resize
        if self.resize is not None:
            self.h, self.w = self.resize
        self.data_set = []
        self.phase = "train"
        self.hard_augs = hard_augs
        # define random crop h and w based 
        # on the original image size: 
        self.random_crop_h = int(self.orig_h*0.9)
        self.random_crop_w = int(self.orig_w*0.9)
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.final_resizing = Resize(self.h, self.w, interpolation=4, p=1.0)
        self.possible_phases = ["train", "val", "test"]

    def set_phase(self, phase, data_set):
        """
        must called after init or to swap training phase 
        """
        self.phase = phase
        if phase not in self.possible_phases:
            raise ValueError('Phase type must be on of: {}'.format(self.possible_phases))
        self.data_set = data_set
        self.transformer = self.get_transforms()
        
    def get_transforms(self):
        list_transforms = []
        if self.phase == "train":
            list_transforms.extend([
                    # Spatial transforms
                    HorizontalFlip(p=0.6),
                    RandomCrop(self.random_crop_h, self.random_crop_w, p=0.8),
                    Resize(self.orig_h, self.orig_w, interpolation=4, p=1.0),
                    # RGB transormations
                    OneOf([
                        RandomBrightness(p=0.5, limit=0.2),
                        RandomContrast(p=0.5, limit=0.2),
                        RandomGamma(p=0.5, gamma_limit=(80, 120))
                    ], p=1.)
                ])
                
            if self.hard_augs:
                list_transforms.extend(
                    # Hard augs 
                    OneOf([
                        RandomFog(p=0.5, fog_coef_lower=0.1, fog_coef_upper=.3, alpha_coef=0.08),
                        RandomRain(p=0.5, slant_lower=-20, slant_upper=20, rain_type=None), # [None, "drizzle", "heavy", "torrestial"]
                        RandomSnow(p=0.5, snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5),
                        RandomSunFlare(p=0.5, flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=3, 
                                        num_flare_circles_upper=6, src_radius=400, src_color=(255, 255, 255))  
                    ], p=0.8)
                )

        list_transforms.append(Normalize(mean=self.mean, std=self.std, p=1))
        list_trfms = Compose(list_transforms)
        return list_trfms

    def __len__(self):
        return len(self.data_set)