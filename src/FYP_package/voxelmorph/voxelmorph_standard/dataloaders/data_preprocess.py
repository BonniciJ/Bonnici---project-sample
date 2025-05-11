import numpy as np
if __package__ is None or __package__ == '': #if this script is run directly in IPython console
    # uses current directory visibility
    import load_deformation as lde #load deformation
else:
    # uses current package visibility
    from . import load_deformation as lde #load deformation

#%%
"""
Module that contains all the pre-processing functions for the data.
"""
#%%
def cal_appendix(pathM):
    """
    Appendix is the size of the crop, such that areas of the image where phantom 
    was never generated do not enter the frame.
    """

    XX_deform, ZZ_deform, XX, ZZ, dx, dz = lde.give_deformation(pathM)
    
    theta = np.concatenate((XX_deform/dx,ZZ_deform/dz)) #in pixels
    
    norm = np.linalg.norm(theta, axis = 0)
    
    crop = np.ceil(np.amax(norm)).astype(int)
        
    return crop

#%% Functions for preprocessing the data
def crop_extras(image, appendix):
    """
    Crops the image, such that areas of the image where phantom was never generated
    do not enter the frame.
    
    #!!! Will require changes to work in 3D.
    
    Parameters
    ----------
    image: ndarray
           square image, with resolution N_pixels * N_pixels
        
    appendix: int
        crop size
              
    Return
    ------
    image: ndarray
           cropped image
    """
    return image[..., appendix:-(appendix),appendix:-(appendix)]


def pad_image(image, factor, pad_cval):
    """
    Pads the image's -2 and -1 axis, increasing them until they are a factor 
    of 'factor'.
    
    Parameters
    ----------
    image: (...,n,m) ndarray
            image to be padded, where 'n', 'm' are the dimensions of the image

    factor: int
            The dimensions of the images will be increased until both -2 and -1
            are multiples of 'factor'.
            
            Default: 16 - current implementation of rthetanet reduces the original
            dimesion by a factor of 16 at the deepest level. Thus, the image
            must be divisible by 16.
            
    pad_cval: float
            Value that the pad has
    """
    
    dims = len(image.shape)
    
    assert dims>=2, f'Error. pad_image accepts images with dim >= 2. Provided image with dim={dims}.'
    
    shape = np.shape(image)
    
    N = shape[-2]
    M = shape[-1]
    
    def give_pads(n,f):
        if n%f !=0:
            n = f - n%f
        else:
            return 0, 0
        
        n1 = n//2
        if n%2 != 0:
            n2 = n-n1
        else:
            n2 = n1
            
        return int(n1), int(n2) 
    
    N1,N2 = give_pads(N,factor)
    M1,M2 = give_pads(M,factor)    
    
    pads = [[0,0]] * (dims-2) + [[N1,N2], [M1, M2]]
    
    image = np.pad(image, pads, constant_values = pad_cval, mode = 'constant')
    
    return image#, pads

def _pad_dim(img, dim, amount, pad_cval):
    dims = len(img.shape)
    pads = [[0,0]] * dims
    
    if amount % 2 == 0: #if even
        amount = int(amount/2)
        pads[dim] = [amount, amount]
    else:
        amount = amount//2
        pads[dim] = [amount, amount + 1]
        
    img = np.pad(img, pads, constant_values = pad_cval, mode = 'constant')
    
    return img

def _crop_dim(img, dim, amount):
    dims = len(img.shape)
    crops = [slice(None)] * dims
    
    if amount % 2 == 0: #if even
        amount = int(amount/2)
        crops[dim] = slice(amount, -amount)
    else:
        amount = amount//2
        crops[dim] = slice(amount, -(amount+1))
    
    crops = tuple(crops)
    
    img = img[crops]
    return img

def resize_image(image, N_pixels_desired, pad_cval):
    """
    Crops the images and the meshes corresponding to their physical locations.

    Parameters
    ----------
    image: (1or2, N, M) or (F, 1or2, N, M) ndarray
            image to be cropped, where 'n' is the size of the image

    N_pixels_desired: int
                    desired pixel size of image
                    
    pad_cval: float
            Value that the pad has
                    
    """
    if type(N_pixels_desired) == int:
        N_pixels_desired = [N_pixels_desired, N_pixels_desired]

    dims = len(image.shape)
    assert dims>=2, f'Error. pad_image accepts images with dim >= 2. Provided image with dim={dims}.'
    shape = np.shape(image)
    H = shape[-2]
    W = shape[-1]
    
    H_d, W_d = N_pixels_desired

    if H_d > H:
        image = _pad_dim(image, -2, H_d-H, pad_cval)
    elif H_d < H:
        image = _crop_dim(image, -2, H-H_d)

    if W_d > W:
        image = _pad_dim(image, -1, W_d-W, pad_cval)
    elif W_d < W:
        image = _crop_dim(image, -1, W-W_d)

    return image

def basic_norm(image):
    img = image - image.min()
    return img/img.max()

def normalise_image(image, US_data_type):
    """
    Normalises images between 0 and 1. Optionally normalises the phase.
    """
    representation = US_data_type['representation']

    if representation == 'bmode':
        dynamic_range = US_data_type['dynamic_range']
        dynamic_range = abs(20 * np.log10(dynamic_range))
        return (image + dynamic_range)/dynamic_range

    elif representation == 'bmodetheta':
        dynamic_range = US_data_type['dynamic_range']
        dynamic_range = abs(20 * np.log10(dynamic_range))
        
        dims = len(image.shape)
        if dims == 4:
            bmode, theta = image[:,0], image[:,1]
        elif dims == 3:
            bmode, theta = image[0], image[1]
        else:
            raise ValueError(f'Dimensions of the provided data are {len(image.shape)}, while expected (F,C,H,W) or (C,H,W), where F is the number of frames.')
        
        bmode = (bmode + dynamic_range)/dynamic_range
        
        if US_data_type['normalise_phase']:
            theta = (theta + np.pi)/(2*np.pi)
        
        return np.stack((bmode, theta), axis = dims-3)


    elif representation == 'envelope':
        return basic_norm(image)
    
    elif representation == 'aib':
        if 'magnitude_normalisation' in US_data_type: #backward compatibility
            if US_data_type['magnitude_normalisation']:
                dims = len(image.shape)
                
                if dims == 4:
                    a, b = image[:,0], image[:,1]
                elif dims == 3:
                    a, b = image[0], image[1]
                else:
                    raise ValueError(f'Dimensions of the provided data are {len(image.shape)}, while expected (F,C,H,W) or (C,H,W), where F is the number of frames.')
                r = (a**2 + b**2)**0.5 
                out = image/r.max() #Magnitude is now between 0 and 1 but any real and imaginary are in ranges [-1,1]
                out = (out + 1)/2 #Real and imaginary now in ranges [0, 1]
                return out
            
        else:
            return basic_norm(image)
    
    elif representation in ['rtheta', 'rcostheta']:
        dims = len(image.shape)
        if dims == 4:
            r, theta = image[:,0], image[:,1]
        elif dims == 3:
            r, theta = image[0], image[1]
        else:
            raise ValueError(f'Dimensions of the provided data are {len(image.shape)}, while expected (F,C,H,W) or (C,H,W), where F is the number of frames.')

        r = basic_norm(r)
        
        if US_data_type['normalise_phase']:
            theta = (theta + np.pi)/(2*np.pi)
        
        return np.stack((r, theta), axis = dims-3)            

#%% used for XX and ZZ mesh preprocessing
def _find_start(large_arr, small_arr):
    """
    If small array is a subset of larger array, finds the index i, at which 
    small array starts inside the larger array
    """
    if len(large_arr) == len(small_arr):
        return 0
    
    n = 0
    while True:
        if n >= len(large_arr):
            raise ValueError('Small array is not inside the large array.')
            
        if np.allclose(large_arr[n:n+len(small_arr)], small_arr, atol=1e-04):
            return n
        n += 1

def _find_non_zero_line(arr, axis):
    if axis == 1:
        temp = arr[0,0,:]
    elif axis == 2:
        temp = arr[0,:,0]
    else:
        raise ValueError('Function not implemented for axis other than 1 and 2.')
    
    i = 0
    while sum(abs(temp)) == 0:
        i += 1
        if axis == 1:
            temp = arr[0,i,:]
        elif axis == 2:
            temp = arr[0,:,i]
    return temp

def _new_line(x_pp, x_og, dx):
    if np.isclose(np.sum(abs(np.diff(np.diff(x_pp)))), 0): #if non-zero means no padding took place along the line
        return x_pp #_case 1, 4
    
    if len(x_pp) > len(x_og):
        x = np.arange(len(x_pp)) * dx
        nonzero = np.nonzero(x_pp)[0]
        if len(nonzero) == len(x_og): #case 5
            i = _find_start(x_pp, x_og)
            x = x - x[i] + x_og[0]
            return x 
        elif len(nonzero) < len(x_og): #case 6
            x_pp = x_pp[nonzero]
            i = _find_start(x_og, x_pp)
            x = x - x[nonzero[0]] + x_og[i]
            return x 
        else: 
            raise ValueError('Post processed x has larger number of non-zero values than the original x. This should not be possible with the way data is being processed.')
    else: 
        pp_i = np.nonzero(x_pp)[0][0]
        pp_start = x_pp[pp_i]
        og_i = np.where(x_og == pp_start)[0][0]
        i = og_i - pp_i
        x = x_og[i: i + len(x_pp)]
        return x #_case 2, 3


#%% Loading and preprocessing the data 
# def preprocess_image(image, appendix, US_data_type, N_pixels_desired, network_reduction_factor, pad_cval):
#     raise ValueError('Verify the image shape that works correcty with this function')
#     image = crop_extras(image, appendix)
#     image = normalise_image(image, US_data_type)
#     if N_pixels_desired:
#         image = resize_image(image, N_pixels_desired = N_pixels_desired, pad_cval = pad_cval)
#     else:
#         image = pad_image(image, factor = network_reduction_factor, pad_cval = pad_cval)
#     return image

def preprocess_movie(frames_dic, appendix, US_data_type, N_pixels_desired, network_reduction_factor, pad_cval):
    shp = frames_dic[0].shape
    frames_array = np.zeros((len(frames_dic), *shp))

    
    #this if statement allows us to load just a single frame JAB
    if len(frames_dic) == 1:
        frames_array = frames_dic
    else:
        for f_idx in frames_dic:
            frames_array[f_idx] = frames_dic[f_idx]
        frames_array = crop_extras(frames_array, appendix)

    frames_array = normalise_image(frames_array, US_data_type)
    
    if N_pixels_desired:
        frames_array = resize_image(frames_array, N_pixels_desired = N_pixels_desired, pad_cval = pad_cval)
    else:
        frames_array = pad_image(frames_array, factor = network_reduction_factor, pad_cval = pad_cval)
        
    return frames_array

def preprocess_deform(XX, appendix, N_pixels_desired, network_reduction_factor, pad_cval_deform):
    XX = crop_extras(XX, appendix)
    if N_pixels_desired:
        XX = resize_image(XX, N_pixels_desired = N_pixels_desired, pad_cval = pad_cval_deform)
    else:
        XX = pad_image(XX, factor = network_reduction_factor, pad_cval = pad_cval_deform)
    return XX

def preprocess_meshes(XX, ZZ, appendix, N_pixels_desired, network_reduction_factor, dx, dz):
    """
    Parameters
    ----------
    XX, ZZ: np.ndarray with shapes (1, H, W)   

    Returns
    -------
    XX, ZZ: np.ndarray with shapes (1, H, W)
        updated XX, ZZ. Instead of padding, in case it is necesssary, this function
        extends XX and ZZ.
    """
    x_og, z_og = XX[0,0,:], ZZ[0,:,0]
    assert (x_og[-1] > x_og[0]), f'x needs to be increasing. Provided x: x{x_og}'
    assert (z_og[-1] > z_og[0]), f'z needs to be increasing. Provided z: {z_og}'
    
    XX = preprocess_deform(XX = XX, appendix=appendix, N_pixels_desired=N_pixels_desired, network_reduction_factor=network_reduction_factor, pad_cval_deform=0)
    ZZ = preprocess_deform(XX = ZZ, appendix=appendix, N_pixels_desired=N_pixels_desired, network_reduction_factor=network_reduction_factor, pad_cval_deform=0)
    
    x_pp = _find_non_zero_line(XX, 1)
    z_pp = _find_non_zero_line(ZZ, 2)
    
    x = _new_line(x_pp, x_og, dx)
    z = _new_line(z_pp, z_og, dz)
        
    XX, ZZ = np.meshgrid(x,z)
    XX = XX[np.newaxis]
    ZZ = ZZ[np.newaxis]
    
    return XX, ZZ

#%% unit tests
def _case1(x_og, dx): #og larger, pp no padding
    x_correct = x_og[4:10]
    x_pp = x_correct.copy()
    out = _new_line(x_pp, x_og, dx)
    assert np.allclose(out, x_correct), f'Case 1 failed. \n correct: \n {x_correct} \n out: \n {out}'

def _case2(x_og, dx): #og larger, pp with padding
    x_correct = x_og[4:10]
    x_pp = x_correct.copy()
    x_pp[0] = 0
    x_pp[1] = 0
    x_pp[-1] = 0
    out = _new_line(x_pp, x_og, dx)
    assert np.allclose(out, x_correct), f'Case 2 failed. \n correct: \n {x_correct} \n out: \n {out}'

def _case3(x_og, dx): #og same, pp with padding
    x_correct = x_og.copy()
    x_pp = x_correct.copy()
    x_pp[0] = 0
    x_pp[1] = 0
    x_pp[-1] = 0
    out = _new_line(x_pp, x_og, dx)
    assert np.allclose(out, x_correct), f'Case 3 failed. \n correct: \n {x_correct} \n out: \n {out}'

def _case4(x_og, dx): #og same, no padding
    x_correct = x_og.copy()
    x_pp = x_correct.copy()
    out = _new_line(x_pp, x_og, dx)
    assert np.allclose(out, x_correct), f'Case 4 failed. \n correct: \n {x_correct} \n out: \n {out}'

def _case5(x_og, dx): #og same, with padding
    x_correct = x_og.copy()
    x_correct = np.pad(x_correct, (1,2))
    x_correct[0] = x_correct[1] - dx
    x_correct[-2] = x_correct[-3] + dx
    x_correct[-1] = x_correct[-2] + dx
    
    x_pp = x_og.copy()
    x_pp = np.pad(x_pp, (1,2))
    out = _new_line(x_pp, x_og, dx)
    assert np.allclose(out, x_correct), f'Case 5 failed. \n correct: \n {x_correct} \n out: \n {out}'

def _case6(x_og, dx): #og smaller, pp with non zero values smaller than og, with padding
    x_correct = x_og.copy()
    x_correct = np.pad(x_correct, (1,2))
    x_correct[0] = x_correct[1] - dx
    x_correct[-2] = x_correct[-3] + dx
    x_correct[-1] = x_correct[-2] + dx
    
    x_pp = x_correct.copy()
    x_pp[:4] = 0
    x_pp[10:] = 0
    out = _new_line(x_pp, x_og, dx)
    assert np.allclose(out, x_correct), f'Case 5 failed. \n correct: \n {x_correct} \n out: \n {out}'

def _test_cases_with_og(x_og, dx):
    _case1(x_og, dx)
    _case2(x_og, dx)
    _case3(x_og, dx)
    _case4(x_og, dx)
    _case5(x_og, dx)
    _case6(x_og, dx)

def _unit_test():
    dx = 0.3
    x_og = np.arange(-10,4,dx)
    _test_cases_with_og(x_og, dx)
    
    x_og = np.arange(0,14,dx)
    _test_cases_with_og(x_og, dx)

    x_og = np.arange(3,17,dx)
    _test_cases_with_og(x_og, dx)

    x_og = np.arange(-14,0,dx)
    _test_cases_with_og(x_og, dx)

    x_og = np.arange(-17,-14,dx)
    _test_cases_with_og(x_og, dx)

if __name__ == '__main__':
    _unit_test()