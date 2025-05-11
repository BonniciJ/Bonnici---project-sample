import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class PairsDataset(Dataset):

    """
        Returns pairs of frames from one movie.
        Pairs are calculated from the middle frame. 
        Use shuffle to get random pairs.
    """


    def __init__(self, path):
        super().__init__()

        self.path = path

        self.movieLength = 25
        self.num_movies = len(os.listdir(path))

        self.pairing = self._pairsFromMiddle(12, 25)


    def __len__(self):
        return self.num_movies * len(self.pairing) * 2 # *2 as we go forward and backward
    
    def __getitem__(self, index):
        backward = index % 2 == 1
        movie = index // (len(self.pairing)  * 2)
        index = index // 2
        pair = self.pairing[index % len(self.pairing)]

        fixed = np.load(self.path + '/' + str(movie) + '/' + str(pair[0]) + '.npy')
        moving = np.load(self.path + '/' + str(movie) + '/' + str(pair[1]) + '.npy')

        fixed = torch.from_numpy(fixed)
        moving = torch.from_numpy(moving)

        # Swap fixed and moving frames if we are going backward
        if backward:
            fixed, moving = moving, fixed

        return fixed, moving                
    

    def _pairsFromMiddle(self, middle_frame, movie_length, include_self_pair=False):
        pairs = [[middle_frame, f] for f in range(0, movie_length)]

        if include_self_pair == False:
            pairs.remove([middle_frame, middle_frame])

        return pairs