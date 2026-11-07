from utils.config import *
import random
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, ColorJitter

aug_ops = [
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
] if AUGMENT else []

transform_train = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
    *aug_ops,
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

class JointTransform:
    """Applies identical geometric transforms to image and mask."""
    def __init__(self, image_size, train=False):
        self.image_size = image_size
        self.train = train
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2)  # image only

    def __call__(self, image, mask=None):
        image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BICUBIC)
        if mask is not None:
            mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.BICUBIC)

        if self.train and AUGMENT:
            if random.random() < 0.5:
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask)

            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BICUBIC)
            if mask is not None:
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.BICUBIC)

            image = self.color_jitter(image)

        image = TF.to_tensor(image)
        if mask is not None:
            mask = TF.to_tensor(mask)
            return image, mask
        return image

class XrayDataset(Dataset):
    """Returns (image, label), or (image, mask, label) when mask_path exists.
    """
    def __init__(self, df, transform=None, unet_model=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.unet_model = unet_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if MASK_ONLY:
            mask_path = row.get("mask_path", None)
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.transform(mask)
            mask_tensor = (mask_tensor > 0.5).float()
            mask_tensor = mask_tensor.repeat(3, 1, 1)

            return mask_tensor, row["label"]
            
        image = Image.open(row["image_path"]).convert("RGB")
    
        mask_path = row.get("mask_path", None)
        if isinstance(mask_path, str) and mask_path:
            mask = Image.open(mask_path).convert("L")
                
            if isinstance(self.transform, JointTransform):
                image_tensor, mask_tensor = self.transform(image, mask)
                   
            else:
                image_tensor = self.transform(image)
                mask_tensor = transforms.ToTensor()(mask)
    
            return image_tensor, mask_tensor, row["label"]
        image_tensor = self.transform(image)
        return image_tensor, row["label"]
