import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import torch.functional as F
from torchvision import transforms

class HeartDataset(Dataset):
    """
    Dataset to load random image pairs from Matthieu's perfusion dataset. 
    """
    def __init__(self, data_folder, transform=None, frames=False, frame_gap=1, num_frames=3, skip_frames=[0, 1, 2, 3]):
        """
        Args:
            image_folder (str): Path to the folder with patient folders.
            transform (callable, optional): Optional transform to be applied
                on an image.

            frames (bool, optional): True if you want frames, otherwise gives pairs
        """
        self.data_folder = data_folder
        self.transform = transform

        self.patient_nums = os.listdir(data_folder)
        self.num_patients = len(self.patient_nums)

        # Fills in number of images in each file for each patient
        # Shape: (# patients, 6)
        self.num_imgs = np.zeros((self.num_patients, 6), dtype=np.int32) # Note, number of 'files' per patient is usually 6, one of them is 3

        for p in range(self.num_patients):
            patient_folder = f"{data_folder}/{self.patient_nums[p]}"
            file_nums = os.listdir(patient_folder)
            nf = len(file_nums)

            for f in range(nf):
                file_folder = f"{patient_folder}/{file_nums[f]}"
                ni = len(os.listdir(file_folder))
                self.num_imgs[p, f] = ni


        # Gets the list of all possible combinations (ordered)
        # Change to _get_pairs_idxs for pairs
        if frames:
            self.sample_idxs = self._get_frame_idxs(num_frames=num_frames, skip_frames=skip_frames)
        # If frames is false, get pairs of images with a gap of frame_gap
        else:
            self.sample_idxs = self._get_pairs_idxs(frame_gap=frame_gap)
        

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # # Cache may not be needed as only loading jpgs. also with so many images it may not be used often
        # self.cache = {}
        # self.MAX_CACHE_SIZE = 100000

         
    def __len__(self):
        # Number of combinations of frames
        return len(self.sample_idxs)
 
    def __getitem__(self, idx): 
        frames_idxs = self.sample_idxs[idx]    
        patient = frames_idxs[0] 
        file = frames_idxs[1]
        frames = frames_idxs[2]

        sample = []
        for frame in frames:
            img = Image.open(f'{self.data_folder}/{self.patient_nums[patient]}/file{file}/frame{frame}.jpg')
            sample.append(self.transform(img))
        
        #sample = torch.stack(sample)
        return sample

        # # If movie not in chache, load it
        # if (patient, file, tuple(frames)) not in self.cache:
        #     # evict an old movie if cache is full:
        #     if len(self.cache) >= self.MAX_CACHE_SIZE:
        #         evict_key = next(iter(self.cache))
        #         del self.cache[evict_key]
        #     # open image and convert to tensor, this is slow...
        #     sample = []
        #     for frame in frames:
        #         img = Image.open(f'{self.data_folder}/patient{patient}/file{file}/frame{frame}.jpg')
        #         sample.append(self.transform(img).squeeze())
            
        #     sample = torch.stack(sample)
        #     self.cache[(patient, file, tuple(frames))] = sample
        #     return sample
        # else:
        #     return self.cahce[(patient, file, tuple(frames))]

        
    def _get_frames_idxs_per_file(self, movieLength, n_frames, skip_frames):
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

        return frame_indices
    
    def _get_frame_idxs(self, num_frames=3, skip_frames=[0, 1, 2, 3]):
        frames = []
        for p in range(self.num_patients):
            for f in range(6):
                frames.append([p, f, self._get_frames_idxs_per_file(self.num_imgs[p, f], num_frames, skip_frames)])

        # Unroll so its just one list 
        ouput = [] #List of all possible samples
        for pf in frames:
            for f in pf[2]:
                ouput.append([pf[0], pf[1], f])

        return ouput
    
    
    def _get_pairs_idxs(self, frame_gap=1):
        pairs = []
        for p in range(self.num_patients):
            for f in range(6):
                for img1 in range(self.num_imgs[p, f]-frame_gap):
                    # Get the second image, which is the first image integers between 1 and frame_gap
                    pairs.append([p, f, [img1, img1+frame_gap]])
                    
        return pairs