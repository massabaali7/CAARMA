import collections
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.io import wavfile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
import soundfile as sf

import torchaudio

def load_audio(filename, second=3):
    waveform, sr = torchaudio.load(filename)
    waveform = waveform.squeeze(0)  # Remove channel dimension if mono

    audio_length = waveform.shape[0]

    if second <= 0:
        return waveform.clone()

    length = int(sr * second)

    if audio_length <= length:
        shortage = length - audio_length
        waveform = torch.nn.functional.pad(waveform, (0, shortage), mode='reflect')
    else:
        start = int(random.random() * (audio_length - length))
        waveform = waveform[start:start + length]

    return waveform.clone()


class Train_Dataset(Dataset):
    def __init__(self, train_csv_path, second=3, do_augmentation=False, augmentation=None, **kwargs):
        self.second = second

        df = pd.read_csv(train_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values
        self.labels, self.paths = shuffle(self.labels, self.paths)
        self.augmentation = augmentation
        self.do_augmentation = do_augmentation

        print("Train Dataset load {} speakers".format(len(set(self.labels))))
        print("Train Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        waveform_1 = load_audio(self.paths[index], self.second)
        waveform_length = waveform_1.shape[-1]
        if self.do_augmentation:
            waveform_1 = self.augmentation(waveform_1)
        
        sample = {
        'waveform':  waveform_1,
        'path': self.paths[index],
        'mapped_id': self.labels[index],
        'lens': waveform_length  # Add the waveform length to the sample
        }
        return sample

    def __len__(self):
        return len(self.paths)
    
    def collate_fn(self, batch):
        audios = [item['waveform'].squeeze(0) for item in batch]
        mapped_ids = [item['mapped_id'] for item in batch]
        mapped_ids = torch.tensor(mapped_ids)
        waveform_lengths = [item['lens'] for item in batch]  # Collect lengths
        waveform_lengths = torch.tensor(waveform_lengths)
        audio_paths = [item['path'] for item in batch]

        audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

        return {
            "waveform": audios_padded,
            "mapped_id": mapped_ids,
            "lens": waveform_lengths,  # Return the lengths as part of the collate function
            "path": audio_paths,
        }
    
    
class Evaluation_Dataset(Dataset):
    def __init__(self, paths, root):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.paths = paths
        self.root = root
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        
        waveform  = load_audio(self.root + self.paths[idx], -1)
        sample = {
            'waveform': torch.FloatTensor(waveform),
            'path': os.path.join(self.root, self.paths[idx]),
            'lens': waveform.shape  # Add the waveform length to the sample
        }
        
        return sample

    def collate_fn(self, batch):
        audios = [item['waveform'] for item in batch]
        waveform_lengths = [item['lens'] for item in batch]  # Collect lengths
        waveform_lengths = torch.tensor(waveform_lengths)
        audio_paths = [item['path'] for item in batch]

        audios_padded = torch.Floattensor(audios)

        return {
            "waveform": audios_padded,
            "lens": waveform_lengths,  # Return the lengths as part of the collate function
            "path": audio_paths
        }
