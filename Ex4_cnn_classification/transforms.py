""" PyTorch interface to our dataset """
import torchvision.transforms as transforms
#from torchvision.transforms import v2


def get_transforms_train():
    """Return the transformations applied to images during training.
    
    See https://pytorch.org/vision/stable/transforms.html for a full list of 
    available transforms.
    """
    transform = transforms.Compose(
        [
            #transforms.RandomResizedCrop(size=(256, 256)),
            #v2.RandomHorizontalFlip(),
            #v2.RandomVerticalFlip(),
            #v2.RandomRotation(degrees=30),
            #v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            #transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ]
    )
    return transform


def get_transforms_val():
    """Return the transformations applied to images during validation.

    Note: You do not need to change this function 
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ]
    )
    return transform