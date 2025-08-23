from .fbanks import Mel_Spectrogram


def build_feature(config):
    if config['features'] == 'Fbank':
        features = Mel_Spectrogram()        
    else:
        raise NotImplementedError    
    
    return features
        