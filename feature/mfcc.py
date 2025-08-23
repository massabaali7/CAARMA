import torchaudio
import torchaudio.transforms as transforms

class MFCC:
    def __init__(self, n_mfcc=13, n_mels=40, f_min=0.0, f_max=None, sample_rate=16000):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.mfcc_transform = transforms.MFCC(
        sample_rate=self.sample_rate,
        n_mfcc=self.n_mfcc,
        melkwargs={
            "n_mels": self.n_mels,
            "f_min": self.f_min,
            "f_max": self.f_max or self.sample_rate / 2,
        },
        )

    def forward(self, waveform):   
        # Extract embeddings and tokens
        mfcc_batch = self.mfcc_tranform(waveform)
        return mfcc_batch
    