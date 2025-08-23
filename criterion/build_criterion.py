from criterion.amsoftmax_mix_gan import amsoftmax_gan

def build_criterion(config):
    if config['criterion'] == 'AMSoftmaxGAN':
        criterion = amsoftmax_gan(embedding_dim=192, num_classes=1211, m=0.2, s=30)
    else:
        raise NotImplementedError

    return criterion