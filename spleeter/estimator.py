import torch
from torch import nn
import math
from torchaudio.functional import istft
import torch.nn.functional as F
from .util import tf2pytorch

from .unet import UNet


def pad_and_partition(tensor, T):
    """
    pads zero and partition tensor into segments of length T

    Args:
        tensor(Tensor): BxCxTxF
    """
    new_len = math.ceil(x.size(2)/T) * T
    tensor = F.pad(tensor, [0, 0, new_len - x.size(2),0])
    [b, c, t, f] = tensor.shape
    split = new_len // T
    return torch.cat(torch.split(tensor, split, dim=2), dim=0)

class Estimator(nn.Module):
    def __init__(self, num_instrumments, check_point):
        super(Estimator, self).__init__()

        # stft config

        self.F = 2048
        self.win_length = 4096
        self.hop_length = 1024
        self.win = torch.hann_window(self.win_length)


        ckpts = tf2pytorch(check_point, num_instrumments)

        # filter
        self.instruments = nn.ModuleList()
        for i in range(num_instrumments):
            print('load instrumment {}'.format(i))
            net = UNet(2)
            ckpt = ckpts[i]
            import pdb; pdb.set_trace()
            net.load_state_dict(ckpt)
            self.instruments.append(net)

    def compute_stft(self, wav):
        # waw is stereo wavs
        # p = (self.n_fft - self.hop_length) // 2
        # wav = F.pad(wav, (p, p), 'reflect')
        stft = torch.stft(
            wav, self.win_length, hop_length=self.hop_length, window=self.win)
        return stft
        # return stft[:, :self.F, :, :].detach()

    def inverse_stft(self, stft):
        """stft_matrix = (batch, freq, time, complex) 
    
            All based on librosa
                - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
            What's missing?
                - normalize by sum of squared window --> do we need it here?
                Actually the result is ok by simply dividing y by 2. 
            """
        pad = self.win_length //2 + 1- stft.size(1)
        stft = F.pad(stft, (0,0, 0, 0, 0, pad))
        wav = istft(stft, self.win_length, hop_length=self.hop_length,
                    window=self.win).unsqueeze(-1)
        return wav.detach()

    def separate(self, wav):
        masks = []
        stft = self.compute_stft(wav)
        for net in self.instrumments:
            mask = net(stft)
            masks.push(mask)

        # normalize mask
        mask_sum = torch.sum([m ** 2 for m in masks])
        wavs = []
        for mask in masks:
            mask = (mask **2)/(mask_sum + 1e-8)
            stft_masked = stft * mask
            wav_masked = self.inverse_stft(stft_masked)
            wavs.append(wav_masked)
        
        return wavs

        

    def forward(self, wav):
        stft = self.compute_stft(wav)
        mask = self.mask_vocal(stft)
        wav, vocal = self.seperate(wav, mask)
