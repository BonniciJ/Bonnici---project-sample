import torch
import torch.nn.functional as F
import numpy as np
import math


class Combined_NCC:

    def loss(self, input, output):
        ncc = NCC()

        target = input[:, 1:, :, :].unsqueeze(2)
        predicted = output[:, :-1, :, :].unsqueeze(2)
        
        B, T1, C, H, W = target.shape
        target = target.flatten(start_dim=0, end_dim=1)   # [B*(T-1), 1, H, W]
        predicted = predicted.flatten(start_dim=0, end_dim=1)  # [B*(T-1), 1, H, W]

        # Compute the NCC loss across all frame pairs
        total_loss = ncc.loss(target, predicted)  # should return a scalar (averaged over windows and samples)

        return total_loss

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win 

        # compute filters
        sum_filt = torch.ones([1, 1, *win])
        if y_true.is_cuda:
            sum_filt = sum_filt.cuda()

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class Combined_MSE:

    def loss(self, input, output):
        mse = MSE()

        target = input[:, 1:, :, :].unsqueeze(2)
        predicted = output[:, :-1, :, :].unsqueeze(2)
        
        B, T1, C, H, W = target.shape
        target = target.flatten(start_dim=0, end_dim=1)   # [B*(T-1), 1, H, W]
        predicted = predicted.flatten(start_dim=0, end_dim=1)  # [B*(T-1), 1, H, W]

        # Compute the NCC loss across all frame pairs
        total_loss = mse.loss(target, predicted)  # should return a scalar (averaged over windows and samples)

        return total_loss


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class BendingEnergy:
    """
    Bending energy regularisation.

    Assumes input of shape (B, C, H, W)
    """    

    def _dx(self, f):
        """Note: output shape will be (B, C, H, W-2), so it will not be square anymore"""
        return (f[:, :, :, 2:] - f[:, :, :, :-2]) / 2
    
    def _dy(self, f):
        """Note: output shape will be (B, C, H-2, W), so it will not be square anymore"""
        return (f[:, :, 2:, :] - f[:, :, :-2, :]) / 2
    
    def _dxy(self, f):
        return self._dx(self._dy(f))
    
    def _dx2(self, f):
        return f[:, :, 1:-1, 2:] - 2*f[:, :, 1:-1, 1:-1] + f[:, :, 1:-1, :-2]

    def _dy2(self, f):
        return f[:, :, 2:, 1:-1] - 2*f[:, :, 1:-1, 1:-1] + f[:, :, :-2, 1:-1]

    def loss(self, _, f):
        """ Uses convolutions to find finite differences and hence bending energy"""

        d2fdx2 = self._dx2(f)
        d2fdy2 = self._dy2(f)
        d2fdxy = self._dxy(f)

        return torch.mean(torch.sum(d2fdx2 ** 2 + 2 * d2fdxy ** 2 + d2fdy2 ** 2, (2, 3)))

