import torch
import random

def panoramic_horizontal_flip(image, p=0.5):
    if random.random() < p:
        return torch.flip(image, [3])  # 水平翻转
    return image

def panoramic_shift(image, max_shift=0.1):
    _, _, _, W = image.shape
    shift = int(random.uniform(-max_shift, max_shift) * W)
    return torch.roll(image, shifts=shift, dims=3)

def panoramic_augmentation(image):
    image = panoramic_horizontal_flip(image)
    image = panoramic_shift(image)
    return image