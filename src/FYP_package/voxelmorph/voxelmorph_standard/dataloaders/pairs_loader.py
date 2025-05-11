from . import sampler as oldsampler
from . import IQ_DAS as lim #load image
from . import data_preprocess as dpp
from . import load_deformation as lde

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy


class PairsDataset(Dataset):

    # This needs some refining, can just use the sampler function to cut down

    def __init__(self, path, dynamic_range=0.01):
        self.path = path
        self.US_data_type = {'representation': 'bmode', 'dynamic_range': dynamic_range}

        self.movieDirs = os.listdir(path)

        self.movieLength = 25
        self.pairsPerMovie = 24

        self.samples = np.zeros((len(self.movieDirs), self.pairsPerMovie, 2), dtype=np.int32)  # array of size (number of movies, number of possible pairs per movie, 2)

        # fill in all possible pairs, using pairs from middle

        
        for i in range(len(self.movieDirs)):
            x = self.pairsFromMiddle(12, 25)
            #random.shuffle(x) pytorch does this
            self.samples[i] = x
        

    def __len__(self):
        return self.samples.shape[0] * self.samples.shape[1]
    
    def __getitem__(self, idx):
        movie = int(idx / self.samples.shape[1])
        sample_idx = idx % self.samples.shape[1]
        frame_pair_idx = self.samples[movie, sample_idx]

        # load in the frame pair
 
        movie_path = self.path + '/' + self.movieDirs[movie] + '/' 
         
        fixed, moving, XX_deform, ZZ_deform, dx, dz = self.sampler(movie_path, frame_pair_idx[0], frame_pair_idx[1], US_data_type = self.US_data_type, N_pixels_desired = 320, network_reduction_factor = 16, pad_cval_image = -1, pad_cval_deform = 0)

        fixed = torch.tensor(fixed[0], dtype=torch.float).unsqueeze(0)
        moving = torch.tensor(moving[0], dtype=torch.float).unsqueeze(0)
        XX_deform = torch.tensor(XX_deform[0], dtype=torch.float).unsqueeze(0)
        ZZ_deform = torch.tensor(ZZ_deform[0], dtype=torch.float).unsqueeze(0)


        return fixed, moving, XX_deform, ZZ_deform, dx, dz
    
    def pairsFromMiddle(self, middle_frame, movie_length, include_self_pair=False):
        pairs = [[middle_frame, f] for f in range(0, movie_length)]

        if include_self_pair == False:
            pairs.remove([middle_frame, middle_frame])

        return pairs
    
    def sampler(self, pathM, fixed_idx, moving_idx, US_data_type, N_pixels_desired, network_reduction_factor, pad_cval_image, pad_cval_deform):
        """
        
        Returns all of the frames of the movie along with the deformation as a single sample.
        
        Return
        ------    
        frames: (M, 1or2, H, W)
            where M = number of frames in a movie
        
        scales: dict with [M] keys
            where M = number of frames in a movie
        
        XX_deform, ZZ_deform, XX, ZZ: (1, H, W)
            where M = number of frames in a movie. 
            ALL VALUES ARE IN PHYSICAL VALUES (not in pixels)
            
        dx, dz: float
            Physical distance between consequitive pixels
            
        pad_cval_image, pad_cval_deform: float
            Constant value with which the image and deform are padded. Note: mesh is not
            padded as it is extended using bilinear interpolation.
        """

        
        parameters = scipy.io.loadmat(pathM+'parameters.mat')
        max_frames = parameters['frames'][0][0]
        
        appendix = dpp.cal_appendix(pathM)
            
        frames = {}
        scales = {}

        fixed = lim.load_IQ(pathM, fixed_idx, US_data_type = US_data_type)
        moving = lim.load_IQ(pathM, moving_idx, US_data_type = US_data_type)

        fixed = dpp.preprocess_movie(frames_dic = fixed, appendix = appendix, US_data_type = US_data_type, N_pixels_desired = N_pixels_desired, network_reduction_factor = network_reduction_factor, pad_cval = pad_cval_image)
        moving = dpp.preprocess_movie(frames_dic = moving, appendix = appendix, US_data_type = US_data_type, N_pixels_desired = N_pixels_desired, network_reduction_factor = network_reduction_factor, pad_cval = pad_cval_image)
        
        XX_deform, ZZ_deform, XX, ZZ, dx, dz = lde.give_deformation(pathM)
    
        XX_deform = dpp.preprocess_deform(XX_deform, appendix = appendix, N_pixels_desired = N_pixels_desired, network_reduction_factor = network_reduction_factor, pad_cval_deform = pad_cval_deform)
        ZZ_deform = dpp.preprocess_deform(ZZ_deform, appendix = appendix, N_pixels_desired = N_pixels_desired, network_reduction_factor = network_reduction_factor, pad_cval_deform = pad_cval_deform)
        XX, ZZ = dpp.preprocess_meshes(XX, ZZ = ZZ, appendix = appendix, N_pixels_desired = N_pixels_desired, network_reduction_factor = network_reduction_factor, dx = dx, dz = dz)


        return fixed, moving, XX_deform, ZZ_deform, dx, dz