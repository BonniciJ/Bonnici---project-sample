import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms

from . import sampler as oldsampler    # used to get deformation

class TimeSeriesDataset(Dataset):

    """
        Returns sets of time ordered frames.
        Skip frames can change 'frame rate' of series. 
    """


    def __init__(self, path='D:/movies_npy', n_frames=5, skip_frames = [0, 1, 2, 3], get_def=False, def_path='D:/movies'):
        super().__init__()

        self.path = path

        self.movie_length = 25
        self.movie_dirs = os.listdir(path)
        self.num_movies = len(self.movie_dirs)
        
        self.frame_indices = self._get_frame_indices(self.movie_length, n_frames, skip_frames)
        self.samples_per_movie = len(self.frame_indices)
        
        self.movie_cache = {}
        self.MAX_CACHE_SIZE = 100

        self.def_cache = {}
        self.MAX_DEF_CACHE = 500

        self.get_def = get_def
        if get_def:
            self.def_path = def_path
            self.old_dirs = os.listdir(self.def_path) # Used to get deformation


        self.cur_movie_idx = None

        self.transform = transforms.Compose([
            transforms.Resize((256, 256))
        ])



    def __len__(self):
        return self.num_movies * len(self.frame_indices)
    
    def __getitem__(self, index):
        movie_idx = self.movie_dirs[index // self.samples_per_movie]
        sample_idx = index % self.samples_per_movie
        frames_idxs = self.frame_indices[sample_idx]

        # If movie not in chache, load it
        if movie_idx not in self.movie_cache:
            # evict an old movie if cache is full:
            if len(self.movie_cache) >= self.MAX_CACHE_SIZE:
                evict_key = next(iter(self.movie_cache))
                del self.movie_cache[evict_key]
            movie = self._load_movie(movie_idx)
            self.movie_cache[movie_idx] = movie
        else:
            movie = self.movie_cache[movie_idx]
         
        # get the frames
        frames = movie[frames_idxs].squeeze(1)
        frames = self.transform(frames)

        if not self.get_def:
            return frames
        
        
        xx, zz = self._load_deformation(movie_idx)

        if frames_idxs[0] > frames_idxs[1]:
            xx = -xx
            zz = -zz

        

        return frames, xx, zz
    

        

    def _load_movie(self, movie_idx):
        movie = []
        for i in range(self.movie_length):
            movie.append(np.load(self.path + '/' + str(movie_idx) + '/' + str(i) + '.npy'))

        return torch.tensor(np.array(np.stack(movie)))
    
    def _get_frame_indices(self, movieLength, n_frames, skip_frames):

        #get the frame indices
        frame_indices = []

        for sf in skip_frames:
            first_list_ending = n_frames + (n_frames - 1) * sf  # where the first list ends
            num_combinations = movieLength - first_list_ending + 1  # number of combinations
            num_combinations = max(0, num_combinations)  # can't be negative
            for i in range(0, num_combinations): # for each combination
                sample = []
                for j in range(0, n_frames): 
                    sample.append(i+j*(sf+1)) # add the frame index

                frame_indices.append(sample) # add the sample to the list

        # append the reversed list (can go both ways)
        frame_indices += [list(reversed(sample)) for sample in frame_indices]

        random.shuffle(frame_indices)

        return frame_indices
    
    def _load_deformation(self, movie_idx):
        """Returns the deformation of the movie"""

        # first check cache
        if movie_idx in self.def_cache:
            return self.def_cache[movie_idx]

        # otherwise load image
        movie_path = f"{self.def_path}/{self.old_dirs[int(movie_idx)]}/"
        US_data_type = {'representation':'bmode','dynamic_range':0.03}
        _, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = oldsampler.samples(
            movie_path, US_data_type, 320, 16, -1, 0
        )

        x_def = XX_deform/dx
        z_def = ZZ_deform/dz

        # cache it (evict old if needed)
        if len(self.def_cache) >= self.MAX_DEF_CACHE:
            self.def_cache.pop(next(iter(self.def_cache)))
        self.def_cache[movie_idx] = (x_def, z_def)

        return x_def, z_def
