import pandas as pd
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as transforms
from scipy import signal
import random
# import psola


class Augmentation:
    def __init__(self, add_noise=True, add_reverb=True, drop_freq=True, drop_chunk=True,
    noise_csv="noise.csv",
    reverb_csv="rir.csv"):
        self.noise_paths = pd.read_csv(noise_csv)["wav"]
        self.reverb_paths = pd.read_csv(reverb_csv)["wav"]
        self.add_noise = add_noise
        self.add_reverb = add_reverb
        self.drop_freq =  drop_freq
        self.drop_chunk = drop_chunk

    def compute_dB(self, waveform):
        val = max(torch.tensor(0.0), torch.mean(torch.pow(waveform, 2)))
        dB = 10*torch.log10(val+1e-4)
        return dB
    
    def add_real_noise(self, waveform):
        clean_dB = self.compute_dB(waveform)

        idx = np.random.randint(0, len(self.noise_paths))
        noise, sample_rate = torchaudio.load(self.noise_paths[idx])
        noise = torch.tensor(noise, dtype=torch.float64)

        snr = np.random.uniform(15, 25)
        snr = torch.tensor(snr)

        noise_length = noise.shape[-1]
        audio_length = waveform.shape[-1]

        if audio_length >= noise_length:
            shortage = audio_length - noise_length
            noise = torch.from_numpy(np.pad(noise.numpy(), ((0, 0), (0, shortage)), 'wrap'))
        else:
            start = np.random.randint(0, noise_length - audio_length)
            start = torch.tensor(start)
            noise = noise[:, start:start+audio_length]

        noise_dB = self.compute_dB(noise)
        noise = torch.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        updated_waveform = waveform + noise
        return updated_waveform.type_as(waveform)

    def add_reverberate(self, waveform):
        audio_length = waveform.shape[-1]
        idx = np.random.randint(0, len(self.reverb_paths))

        path = self.reverb_paths[idx]
        rir, sample_rate = torchaudio.load(path)
        rir = rir / torch.sqrt(torch.sum(rir ** 2))

        updated_waveform = torch.tensor(signal.convolve(waveform, rir, mode='full'))
        return updated_waveform[..., :audio_length].type_as(waveform)

    def drop_frequency(self, waveform, sample_rate):
        freq_mask_param=15 
        num_masks=1
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        freq_masking = transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        
        for _ in range(num_masks):
            updated_waveform = freq_masking(waveform)

        return updated_waveform.type_as(waveform)
        
          
    def drop_chunk_waveform(self, waveform):
        drop_count_low = 1
        drop_count_high = 3
        drop_length_low = 1000
        drop_length_high = 2000

        dropped_waveform = waveform.clone()
        drop_times = random.randint(drop_count_low, drop_count_high)

        if drop_times == 0:
            return dropped_waveform

        lengths = torch.randint(low=drop_length_low, high=drop_length_high + 1, size=(drop_times,))
        start_min = 0
        start_max = waveform.shape[-1] - lengths.max().item()

        starts = torch.randint(low=start_min, high=start_max + 1, size=(drop_times,))
        ends = starts + lengths

        for j in range(drop_times):
            dropped_waveform[starts[j]:ends[j]] = 0.0

        return dropped_waveform.type_as(waveform)

    # def add_psola(self, waveform):
    #     waveform = waveform.numpy()
    #     waveform = psola.vocode(audio=waveform, )


    def __call__(self, x, sr):
        if self.add_reverb:
            x = self.add_reverberate(x)
        
        if self.add_noise:
            x = self.add_real_noise(x)

        if self.drop_freq:
            x = self.drop_frequency(x, sr)

        if self.drop_chunk:
            x = self.drop_chunk_waveform(x)

        return x
