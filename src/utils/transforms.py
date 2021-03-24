import torch
from . import functional as F
from torchvision.transforms.functional import hflip

class ToCoordColorPair:
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to be converted to (coordinator, color) pair.
        Returns:
            (coordinator, color): of size [W*H, 2], [W*H, 3]
        """
        return F.toCoordColorPair(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class RandomDFlip(torch.nn.Module):
    """D flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.dflip(img)
        return img


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomHorizontalFlipList(torch.nn.Module):
    """D flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            results = []
            for image in images:
                results.append(hflip(image))
            return results
        return images


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)