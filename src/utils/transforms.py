
class ToCoordColorPair:
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    
    def make_coord(shape, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
        """
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def __call__(self, img):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to be converted to (coordinator, color) pair.
        Returns:
            (coordinator, color): of size [W*H, 2], [W*H, 3]
        """
        coord = make_coord(img.shape[-2:])
        rgb = img.view(3, -1).permute(1, 0)
        return coord, rgb

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
            return img.transpose(-2, -1)
        return img


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)