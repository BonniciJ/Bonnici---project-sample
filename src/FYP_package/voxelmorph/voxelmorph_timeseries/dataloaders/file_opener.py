import numpy as np

from torch.utils.data import Dataset

if __package__ is None or __package__ == '': #if this script is run directly in IPython console
    # uses current directory visibility
    import pairs_creator
    import sampler

else:
    # uses current package visibility
    from . import pairs_creator
    from . import sampler
    #from .. import grid_sampler as gs
    import torch.nn.functional as F
    import torch

#%%
"""
Module used to open the frame.mat files and crop them accordingly thereby generating
the dataset. Includes functions for splitting the training data as well as displaying
the images.

Last version developed in Archieve_models/vxm is V10.
Last version developed in Archieve_models/Unet is V11.
Last version developed in Archieve/ML/mynet... is V20


V21: 
To do:

> Seperate data pre-processing components into a different module
> Separate data loading into a different module

> Normalise with mean 0 and std of 1
> Normalising between -1 and 1

> Need to ensure that IQ is dynamic range limited and normalised - make this an option.

> Caching might not be most useful anymore. Perhaps would be better to save data of specific size.

> Provide option for cache-less dataset

> Possibly use torch.tensor where possible as conversion to numpy each time might be time consuming

> Possibly remove zero_pad_images option from dataloader (haven't used it in a while)

Completed:
> Separate this module into different files as some files are quite disconnected.
> Added IQ_demodulated
> Updated using new method of decoding the deformation field
> Rename the existing files from 'movie/image_data/' to 'movie/Bmode'
> Changing the way deform is saved - calculating the largest size (312x312), then resizing according to the needs
> Completed the update described below (referred to as 'in future'):
> There was a bug in the deformation field caused by the presence of np.flip in 
y_span = np.flip(parameters['pixel_zspan'])[0], in give_mesh. This means that the deformation vector meant
to be in [0,a] is located in [-1, a] and vice-versa for all 'a'. This can be overcome in 2 ways. Currently,
implemented deform = np.flip(deform,axis=1) in the dataset. However, in future, must delete
all of the existing deformation fields and recalculate them at the highest resolution (with this bug removed).
Consequently, would need to remove the current solution to this issue.
> There was a bug in give_deformation(). Therefore, the saved deformation fields have to be cleared and updated.
Correct:    deformation = calculate_deformation(pathM, XX, YY, 0, frames_per_movie-1, frames_per_movie)
Incorrect:  deformation = calculate_deformation(pathM, XX, YY, 0, frames_per_movie, frames_per_movie)
> Change the way data is cached - all frames and the GT deformations instead each pair,
then access each individually
> There was a bug in normalisation of the IQ images. This was fixed. Now aib representation
is normalised between 0 and 1, while rtheta and rcostheta have only the r component normalised 
between 0 and 1.
> Changed the way caching is done to be more efficient (should occupy much less space)
> Fixed a bug with how flipping is done - now works for IQ data as well as BMode
> I am not sure that flipping works as expected. After modification it didn't. Originally newaxis was added
at the very end. However, with the the implementation, newaxis is added at the start. Therefore, shape would
be (1,N,N) instead of (N,N). The axis of flipping was axis=1. Thus, it worked for the original images
but not for the next ones. This also highlights that there was a bug with the IQ implementation, since
it's shape is 2,N,N. Thus flipping would happen about the wrong axis.
> Implement IQ demodulation as an option, which would create a new dir if it does not exist
and save new files there
> Changed split_path_list from randomly selecting movies to a non-random selection.
Will faciliate cross-comparison
> Implement automatic deformation field saving for translation, shearing and scaling (I don't think rotations can be saved).
> There is a bug of finding dx in fileopener when desired pixel size ~ 312 (due to the way dx is found)
makes GT_theta_grided = GT/dx divide by zero
> Removing all MotionDataset classes, other than the cached version.
> Make sure to save the deformation field instead of re-calculating it every single time.
Make the function check if the deformation field with all the same parameters exists. If yes, the just load it,
it not, then calculate it and save it. This will make this implementation of the function backward compatible.    

Notes
-----
> Implemented appendable dataset, such that smart training can be used, without losing the cached
data
> The MotionDataset prior to V14 had a bug where only the first 50% of cached 
data was used due to the way ind calculated.
> Included data augemntation in form of flipping images along y axis. For flipping 
images used np.flip(image, axis = 1), where np.shape(image) is (312, 312). For 
deformation field used 
XX_deformation = -np.flip(XX_deformation, axis = 1).copy()
YY_deformation = np.flip(YY_deformation, axis = 1).copy()
note the negative sign for XX as in addition to switching the values around needed 
to also negate the values to for correct implementation of deformation flipping.

> Dataset class outputs the GF deformation field between the image pair
such that fixed = moving + deformation_field. Note: deformation has shape (2,N,N), 
where deformation_field[0] corresponds to X deformation of the image and 
deformation_field[1] corresponds to Y deformation of the image. 

"""

#%% Additional pre-processing
def binarise_image(image):
    """
    If intensity is larger or equal to 0.5 it is set to 1, else it is set to 0.
    """
    return np.where(image<0.5, 0, 1)

from scipy.ndimage import correlate
def _convolve(image, distance):
    kernel_size = distance*2 + 1
    kernel = np.ones((1,kernel_size, kernel_size))
    return correlate(image, kernel, mode='constant', cval = 0)

def _nonzero(image, threshold):
    return np.where(image > threshold, 1., 0.)

def image_to_filter(image, distance, threshold = 0):
    """
    Generates a filter from an image, such that all non-zero values are set to 1.
    If distance > 1, pixels that are ks pixels away from the non-zero values 
    are also set to 1. A pixels is considered a neighbour horizontally, vertically
    and diagonally.
    
    e.g.       distance = 0    distance = 1     distance = 2
    7 0 9 0    1 0 1 0         1 1 1 1          1 1 1 1
    4 0 0 0    1 0 0 0         1 1 1 1          1 1 1 1
    0 0 0 0    0 0 0 0         1 1 0 0          1 1 1 1
    0 0 0 0    0 0 0 0         0 0 0 0          1 1 1 0
    
    Parameters
    ----------
    image: np.ndarray with shape (1, H, W)
    
    distance: int >= 0
    
    threshold: float >=0
    
    Returns
    fil: np.ndarray with shape (1, H, W)
    
    """    
    assert len(image.shape) == 3, f'Expected input shape of the image is 3 dimensional, provided image with {image.shape}'
    # assert np.all(image >= 0), 'All values in the image must be greater or equal to zero.'
    assert (distance >=0) and (type(distance) is int), f'Distance value must be a positive integer, provided distancce = {distance} of type {type(distance)}.'
    
    fil = _nonzero(image, threshold)
    
    fil = _convolve(fil, distance)
    fil = _nonzero(fil, threshold)
    return fil


#%% Defining custom dataset class
class MotionDatasetCached(Dataset):
    """US motion dataset."""
    
    def __init__(self, 
                 pathM_list,
                 config_file):
        
        self.pathM_list = pathM_list
        self.pairing = 1 #config_file['pairing']       
        self.inversed_pairs = True #config_file['inversed_pairs']
        
        self.flipping = False #config_file['flipping']
        self.binarise = False #config_file['binarise']
        
        self.US_data_type = {'representation': 'bmode', 'dynamic_range': 0.01} #config_file['US_data_type']

        self.N_pixels_desired = 320 #config_file['N_pixels_desired']
        self.network_reduction_factor = 1 #config_file['network_reduction_factor']

        #self.transform = config_file['transform']
        self.frames_per_movie = 25 #config_file['frames_per_movie']
        
        
        
        self.pairs = pairs_creator.pairs_creator(pairing = self.pairing, 
                                                 frames_per_movie = self.frames_per_movie, 
                                                 return_inverse_order=self.inversed_pairs)
        self.pairs_n = len(self.pairs)
        
        self._cal_len() 
        
        self.cached_movies = {} #dic of cached samples
        self.cached_idx = [] #list of the cached samples' idx
        
        self.pixelated = True# config_file['pixelated'] #If deformation needs to be pixelated

        self.pad_cval_image = -1 #config_file['pad_cval_image']
        self.pad_cval_deform = 0 #config_file['pad_cval_deform']

       
        # if 'eps' in config_file['loss_name'] or 'ccy' in config_file['loss_name']:
        #     self._make_moved = True
        # else:
        #     self._make_moved = False
        self._make_moved = False

        # if 'pairing_eps' in config_file:
        #     self.pairing_eps = config_file['pairing_eps']

        self.eps_mode = False
        
        # if 'mask_deformation' in config_file: #for backward compatibility
        #     self.mask_deformation = config_file['mask_deformation']
        # else:
        #     self.mask_deformation = None
        self.mask_deformation = None
        
        # if 'float_precision' in config_file: #for backward compatibility
        #     self.float_precision = config_file['float_precision']
        # else:
        #     self.float_precision = 'float64'
        self.float_precision = 'float64'

    def append_dataset(self, pathM_list): #Done
        self.pathM_list += pathM_list        
        self._cal_len()
        
        #Adding new empty entrances
        cache_len = len(self.cache_frames)
        added_len = len(pathM_list)
        for i in range(added_len):
            self.cache_frames[i+cache_len] = {}
            self.cache_deformations[i+cache_len] = {}
        
    def _cal_len(self): #Done
        self.length = self.pairs_n*len(self.pathM_list)
        if self.flipping == True:
            self.length = self.length*2

    def __len__(self): #Done
        return self.length

    ####################################
    # All the functions to do with eps #
    ####################################
    def eps_mode_on(self):
        self.cached_idx = [] #Resetting cause eps and normal mode cache idx are different
        
        self.pairs = pairs_creator.pairs_creator(pairing = self.pairing_eps, 
                                                 frames_per_movie = self.frames_per_movie, 
                                                 return_inverse_order=self.inversed_pairs)
        self.pairs_n = len(self.pairs)
        
        self._cal_len()
        self.eps_mode = True
    
    def eps_mode_off(self):
        self.cached_idx = [] #Resetting cause eps and normal mode cache idx are different
        
        self.pairs = pairs_creator.pairs_creator(pairing = self.pairing, 
                                                 frames_per_movie = self.frames_per_movie, 
                                                 return_inverse_order=self.inversed_pairs)
        self.pairs_n = len(self.pairs)
        
        self._cal_len()
        self.eps_mode = False

    ######################################################
    # All the functions below are to do with __getitem__ #
    ######################################################
    def cal_idx(self, idx):
        """
        Returns ind of movie and pair, as well as flag 'flipped'. Designed such
        that (movie_i, pair_i, flipped) are in the following order:
        (0,0,False), (0,1,False),...,(0,N-1,False), (0,0,True),...,(0, N-1, True), (1,0,False),...,
        where N is self.pairs_n.
        """
        devisor = self.pairs_n
        if self.flipping:
            devisor = devisor * 2

        movie_i = idx//devisor
        pair_i = idx - movie_i*devisor
        
        flipped = False
        if pair_i//self.pairs_n == 1:
            flipped = True
            pair_i = pair_i%self.pairs_n
                
        return movie_i, pair_i, flipped
    
    #########
    # Cache #
    #########
    def _cache_movie(self, movie_i): #Done
        
        if movie_i in self.cached_movies: #avoids need to cache the same movie multiple times     
            return
        
        pathM = self.pathM_list[movie_i]
        frames, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = sampler.samples(pathM, self.US_data_type, self.N_pixels_desired, self.network_reduction_factor, self.pad_cval_image, self.pad_cval_deform)
        
        if self.binarise:
            for i in range(len(frames)):
                frames[i] = binarise_image(frames[i])
        
        if self.pixelated:
            XX_deform = XX_deform/dx
            ZZ_deform = ZZ_deform/dz
        
        if self._make_moved:
            # fixed = frames[0][np.newaxis]
            
            if not self.pixelated:                
                theta = np.concatenate((XX_deform/dx, ZZ_deform/dz))[np.newaxis]
            else:
                theta = np.concatenate((XX_deform, ZZ_deform))[np.newaxis]
                
            moved = np.zeros_like(frames)
            for i in range(len(scales)):
                #moved ~ fixed
                #moved[i] = gs.grid_sampler(img = frames[i][np.newaxis], deformation_field = theta * scales[i])[0]
                #from chatgpt
                # Unsqueeze to add batch and channel dimensions (if not already present)
                #frame = frames[i].unsqueeze(0)  # Add batch dimension
                deformation_field = torch.tensor(theta * scales[i])  # Apply scaling
                deformation_field = deformation_field.permute(0, 2, 3, 1)

                # Perform grid sampling
                # deformation_field should be normalized to the range [-1, 1]
                # deformation_field must have the shape [1, H, W, 2]
                moved[i] = F.grid_sample(torch.tensor(frames[i][np.newaxis]), deformation_field, mode='bilinear', padding_mode='zeros', align_corners=True)[0]
                
            
            
        else:
            moved = None
            # moved = np.zeros_like(frames)
        
        if self.mask_deformation:
            fixed = frames[0]
            
            if self.US_data_type['representation'] == 'aib':
                fixed = (fixed[0]**2 + fixed[1] ** 2)**0.5
                fixed = fixed[np.newaxis]
            elif self.US_data_type['representation'] in ['rtheta', 'bmodetheta']:
                fixed = fixed[0][np.newaxis]
            
            #(1, H, W)
            filt = image_to_filter(fixed, distance = self.mask_deformation['distance'], threshold= self.mask_deformation['threshold'])
        else:
            #(1, H, W)
            filt = None
            # filt = np.ones_like(XX_deform) 
        
        dic = {'frames': frames,
               'scales': scales,
               'XX_deform': XX_deform,
               'ZZ_deform': ZZ_deform,
               'XX': XX,
               'ZZ': ZZ,
               'dx': dx,
               'dz': dz,
               'moved': moved,
               'filt':filt}
        
        for key, val in dic.items():
            if (key != 'scales') and (val is not None):
                dic[key] = val.astype(self.float_precision)
        
        self.cached_movies[movie_i] = dic
        
    def create_cache(self, idx): #Done
        movie_i, _, _ = self.cal_idx(idx)
        
        self._cache_movie(movie_i)
        self.cached_idx.append(idx)

    def _get_from_cache_normal_mode(self, idx):
        movie_i, pair_i, flipped = self.cal_idx(idx)
        
        f_i, m_i = self.pairs[pair_i]

        #Loading cached values
        cache = self.cached_movies[movie_i]     
        f, m = cache['frames'][f_i], cache['frames'][m_i]
        
        if self._make_moved: #if false, moved is None
            moved = cache['moved'][m_i]
        else:
            moved = np.zeros_like(f)
            
        XX_deform, ZZ_deform = cache['XX_deform'], cache['ZZ_deform']
        XX, ZZ = cache['XX'], cache['ZZ']
        XXZZ = np.concatenate((XX, ZZ))
        dx, dz = cache['dx'], cache['dz']
        
        filt = cache['filt']
        if filt is None:
            filt = np.ones_like(XX_deform) 
        
        f_s, m_s = cache['scales'][f_i], cache['scales'][m_i]
        scale = np.abs(m_s-f_s)
        assert scale >= 0, f'Scale must be greater or equal to zero. Actual scale requested: {scale}.'
        
        #Flipping if necessary
        if flipped:
            #.copy() necessary, otherwise has a stride of [::-1], which is not ok for pytorch
            f = np.flip(f, axis = 2).copy() #axis=2 since shape is (1or2,N,N)
            m = np.flip(m, axis = 2).copy()
            XX_deform = -np.flip(XX_deform, axis = 2).copy() #axis=2 since shape is (1,N,N). If it were (N,N), then axis would be =1
            ZZ_deform = np.flip(ZZ_deform, axis = 2).copy()
            
            moved = np.flip(moved, axis = 2).copy()
            filt = np.flip(filt, axis = 2).copy()
            
        deform = scale * np.concatenate((XX_deform, ZZ_deform))
        
        assert len(f.shape)==3, f'Error, output shape of fixed image in file_opener is not (1or2, H, W), it is {f.shape}.'
        assert len(m.shape)==3, f'Error, output shape of moving image in file_opener is not (1or2, H, W), it is {m.shape}.'
        assert len(deform.shape)==3, f'Error, output shape of deformation in file_opener is not (2, H, W), it is {deform.shape}.'
        
        return f, m, deform, XXZZ, dx, dz, moved, filt

    def _get_from_cache_eps_mode(self, idx):
        movie_i, pair_i, flipped = self.cal_idx(idx)
        
        m_i_1, m_i_2 = self.pairs[pair_i]
    
        #Loading cached values
        cache = self.cached_movies[movie_i]     
        m_1, m_2 = cache['frames'][m_i_1], cache['frames'][m_i_2]
        moved_2_0 = cache['moved'][m_i_2]
        
        XX_deform, ZZ_deform = cache['XX_deform'], cache['ZZ_deform']
        XX, ZZ = cache['XX'], cache['ZZ']
        XXZZ = np.concatenate((XX, ZZ))
        dx, dz = cache['dx'], cache['dz']
        
        m_s_1 = cache['scales'][m_i_1]
        m_s_2 = cache['scales'][m_i_2]
        f_s = cache['scales'][0]
        
        scale_1 = m_s_1-f_s
        scale_2 = m_s_2-f_s
        
        assert scale_1 >= 0, f'Scale must be greater or equal to zero. Actual scale requested: {scale_1}.'
        assert scale_2 >= 0, f'Scale must be greater or equal to zero. Actual scale requested: {scale_1}.'
        
        filt = cache['filt']
        if filt is None:
            filt = np.ones_like(XX_deform) 
        
        #Flipping if necessary
        if flipped:
            #.copy() necessary, otherwise has a stride of [::-1], which is not ok for pytorch
            m_1 = np.flip(m_1, axis = 2).copy() #axis=2 since shape is (1or2,N,N)
            m_2 = np.flip(m_2, axis = 2).copy()
            XX_deform = -np.flip(XX_deform, axis = 2).copy() #axis=2 since shape is (1,N,N). If it were (N,N), then axis would be =1
            ZZ_deform = np.flip(ZZ_deform, axis = 2).copy()
            
            moved_2_0 = np.flip(moved_2_0, axis = 2).copy()
            filt = np.flip(filt, axis = 2).copy()
            
        deform_1_0 = scale_1*np.concatenate((XX_deform, ZZ_deform))
        deform_2_0 = scale_2*np.concatenate((XX_deform, ZZ_deform)).copy()
        
        return m_1, m_2, deform_1_0, deform_2_0, XXZZ, dx, dz, moved_2_0, filt

    def get_from_cache(self, idx):
        if self.eps_mode:
            sample = self._get_from_cache_eps_mode(idx = idx)        
        else:
            sample = self._get_from_cache_normal_mode(idx = idx)        
        return sample

    def clear_cache(self):
        #To be used solely during evaluation of the models
        self.cached_movies = {} #dic of cached samples
        self.cached_idx = [] #list of the cached samples' idx

    def __getitem__(self, idx): #Done
        if idx in self.cached_idx:
            sample = self.get_from_cache(idx)
            
        else:
            self.create_cache(idx)
            sample = self.get_from_cache(idx)
                        
        return sample    
    
#%%
def _test_filter():
    image = np.array([[7,0,9,0], [4,0,0,0], [0,0,0,0], [0,0,0,0]])[np.newaxis]
    threshold = 0
    
    distance = 0
    filt_out = image_to_filter(image, distance, threshold)
    filt_correct = np.array([[1,0,1,0], [1,0,0,0], [0,0,0,0], [0,0,0,0]])[np.newaxis]
    assert np.allclose(filt_out, filt_correct), f'distance = {distance} failed. \n filt_correct: \n {filt_correct} \n filt_out: \n {filt_out}'
    
    distance = 1
    filt_out = image_to_filter(image, distance, threshold)
    filt_correct = np.array([[1,1,1,1], [1,1,1,1], [1,1,0,0], [0,0,0,0]])[np.newaxis]
    assert np.allclose(filt_out, filt_correct), f'distance = {distance} failed. \n filt_correct: \n {filt_correct} \n filt_out: \n {filt_out}'
    
    distance = 2
    filt_out = image_to_filter(image, distance, threshold)
    filt_correct = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,0]])[np.newaxis]
    assert np.allclose(filt_out, filt_correct), f'distance = {distance} failed. \n filt_correct: \n {filt_correct} \n filt_out: \n {filt_out}'

if __name__ == '__main__':
    _test_filter()
    print('All tests passed')