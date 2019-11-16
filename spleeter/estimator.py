import torch
from torch import nn
import math
from torchaudio.functional import istft
import torch.nn.functional as F
from .util import tf2pytorch

from .unet import UNet


def load_ckpt(model, ckpt):
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            target_shape = state_dict[k].shape
            assert target_shape == v.shape
            state_dict.update({k: torch.from_numpy(v)})

    model.load_state_dict(state_dict)
    return model


def pad_and_partition(tensor, T):
    """
    pads zero and partition tensor into segments of length T

    Args:
        tensor(Tensor): BxCxTxF
    """
    new_len = math.ceil(tensor.size(2)/T) * T
    tensor = F.pad(tensor, [0, 0,  0, new_len - tensor.size(2)])
    [b, c, t, f] = tensor.shape
    split = new_len // T
    return torch.cat(torch.split(tensor, T, dim=2), dim=0)


class Estimator(nn.Module):
    def __init__(self, num_instrumments, check_point):
        super(Estimator, self).__init__()

        # stft config

        self.F = 1024
        self.T = 512
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

            net = load_ckpt(net, ckpt)

            # net.eval()
            self.instruments.append(net)

    def compute_stft(self, wav):
        # waw is stereo wavs
        # p = (self.n_fft - self.hop_length) // 2
        # wav = F.pad(wav, (p, p), 'reflect')
        stft = torch.stft(
            wav, self.win_length, hop_length=self.hop_length, window=self.win)
        # return stft
        stft = stft[:, :self.F, :, :].detach()
        real = stft[:, :, :, 0].unsqueeze(-1)
        im = stft[:, :, :, 1].unsqueeze(-1)
        mag = torch.sqrt(real ** 2 + im ** 2)
        return stft, mag

    def inverse_stft(self, stft):
        """stft_matrix = (batch, freq, time, complex) 

            All based on librosa
                - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
            What's missing?
                - normalize by sum of squared window --> do we need it here?
                Actually the result is ok by simply dividing y by 2. 
            """
        pad = self.win_length // 2 + 1 - stft.size(1)
        stft = F.pad(stft, (0, 0, 0, 0, 0, pad))
        wav = istft(stft, self.win_length, hop_length=self.hop_length,
                    window=self.win).unsqueeze(-1)
        return wav.detach()

    def separate(self, wav):
        B, L = wav.shape
        masks = []
        stft0, mag0 = self.compute_stft(wav[0].unsqueeze(0))
        stft1, mag1 = self.compute_stft(wav[1].unsqueeze(0))

        stft_mag = torch.cat([mag0, mag1], axis=-1)
        stft_mag = pad_and_partition(stft_mag, self.T)
        stft_mag = stft_mag.permute([0, 3, 2, 1])

        # pad stft0
        new_len = math.ceil(stft0.size(2)/self.T) * self.T
        stft0 = F.pad(stft0, [0, 0,  0, new_len - stft0.size(2)])

    
        for net in self.instruments:
            # net.eval()
            mask = net(stft_mag)
            masks.append(mask)

        # normalize mask
        mask_sum = sum([m ** 2 for m in masks])
        wavs = []
        for mask in masks:
            mask = (mask ** 2)/(mask_sum + 1e-8)
            mask = mask.permute([0, 3, 2, 1])[:,:,:,0].unsqueeze(-1)
            import pdb; pdb.set_trace()
            stft_masked = stft0 * mask
            stft_reshaped = torch.cat(
                torch.split(stft_masked, B, dim=0), dim=2)
            wav_masked = self.inverse_stft(stft_reshaped)
            wavs.append(wav_masked)

        return wavs

    def forward(self, wav):
        stft = self.compute_stft(wav)
        mask = self.mask_vocal(stft)
        wav, vocal = self.seperate(wav, mask)

        # import pdb; pdb.set_trace()
        for net in self.instruments:
            net.eval()
            mask_real = net(stft_real)
            mask_im = net(stft_im)

            # generate only one channel
            mask = torch.cat(
                [mask_real[:, :, :, 0], mask_im[:, :, :, 0]], axis=-1)
            masks.append(mask)

        # normalize mask
        mask_sum = sum([m ** 2 for m in masks])
        import pdb; pdb.set_trace()
        wavs = []
        for mask in masks:
            mask = (mask ** 2)/(mask_sum + 1e-8)
            stft_masked = stft * mask
            # import pdb; pdb.set_trace()
            stft_masked = stft_masked.permute([0, 3, 2, 1])
            stft_reshaped = torch.cat(
                torch.split(stft_masked, B, dim=0), dim=2)
            wav_masked = self.inverse_stft(stft_reshaped)
            wavs.append(wav_masked)

        return wavs

    def forward(self, wav):
        stft = self.compute_stft(wav)
        mask = self.mask_vocal(stft)
        wav, vocal = self.seperate(wav, mask)
