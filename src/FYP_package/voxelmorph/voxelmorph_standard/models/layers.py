import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', padding='zeros'):
        super().__init__()
        self.mode = mode
        self.padding = padding



    def forward(self, x, flo):
        
        """
        Clara
        warp an image/tensor (im2) back to im1, according to the optical flow
    
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
    
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
        yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
        xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
        yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
        grid = torch.cat((xx ,yy) ,1).float()
    
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo
    
        # scale grid to [-1,1]
        vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
        vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0
    
        vgrid = vgrid.permute(0 ,2 ,3 ,1)
        output = nnf.grid_sample(x.float(), vgrid.float(), mode = self.mode ,padding_mode = self.padding, align_corners=True)
    
        return output



class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
