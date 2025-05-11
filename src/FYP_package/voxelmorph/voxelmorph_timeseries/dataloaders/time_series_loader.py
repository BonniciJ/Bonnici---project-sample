from . import sampler as oldsampler

import os
import torch
from torch.utils.data import Dataset
import random


class TimeSeriesDataset(Dataset):

    # Needs to produce a dataset of n frames, where:
        # all frames come from a single movie
        # the frames are in order
        # There can be gaps between frames (i.e. different frame rate)

    # So could have moving window sliding along each movie
    # Can do it in reverse order too

    def __init__(self, path, n_frames=5, skip_frames = [0, 1, 2, 3]):
        self.path = path
        self.US_data_type = {'representation': 'bmode', 'dynamic_range': 0.03}

        self.movieDirs = os.listdir(path)
        self._shuffle_dirs()

        self.movieLength = 25
        self.n_frames = n_frames

        self.frame_indices = self._get_frame_indices(self.movieLength, n_frames, skip_frames)

        self.samples_per_movie = len(self.frame_indices)

        self.current_movie_idx = None
        self.current_movie_frames = None
        self.current_XX_deform = None
        self.current_ZZ_deform = None

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
        

    def __len__(self):
        return len(self.frame_indices) * len(self.movieDirs)
    
    def __getitem__(self, idx):
        movie_idx = idx // self.samples_per_movie
        sample_idx = idx % self.samples_per_movie
        frames_idxs = self.frame_indices[sample_idx]

        # if the movie has changed, load the movie
        if movie_idx != self.current_movie_idx:
            movie_path = self.path + '/' + self.movieDirs[movie_idx] + '/' 
            self._load_movie(movie_path) 
            self.current_movie_idx = movie_idx
            random.shuffle(self.frame_indices) 

            # If we are back at movie 0, shuffle the movie dirs
            if movie_idx == 0:
                self._shuffle_dirs()
         
        # get the frames
        frames = self.current_movie_frames[frames_idxs].squeeze(1)

        return frames, self.current_XX_deform, self.current_ZZ_deform
    
    def get_XX_deform(self):
        return self.current_XX_deform
    
    def get_ZZ_deform(self):
        return self.current_ZZ_deform    


    def _load_movie(self, pathM):
        frames, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = oldsampler.samples(pathM, self.US_data_type, 320, 16, -1, 0)

        self.current_movie_frames = torch.tensor(frames)
        self.current_XX_deform = XX_deform
        self.current_ZZ_deform = ZZ_deform

    def _shuffle_dirs(self):
        random.shuffle(self.movieDirs)





# Some testing code
if __name__ == '__main__':
    dataset = TimeSeriesDataset('D:/movies')
    print(f'Dataset length: {len(dataset)}')  

    d = dataset[1000]

    print(f'Data shape: {d.shape}')  

    # Some plotting code to show one of the samples

    # import matplotlib.pyplot as plt

    # for i in range(d.shape[0]):
    #     plt.subplot(1, d.shape[0], i+1)
    #     plt.imshow(d[i])
    # plt.show()


    # # plot the deformations
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(dataset.get_XX_deform()[0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(dataset.get_ZZ_deform()[0])
    # plt.show

    # plt.figure()


    # #make a graphic like a gif of the frames using matplotlib
    
    # for j in range(100):
    #     for i in range(d.shape[0]):
    #         plt.imshow(d[i])
    #         plt.pause(0.1)
    #         plt.draw()

    #     plt.pause(0.5)

