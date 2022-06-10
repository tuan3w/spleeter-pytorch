import numpy as np
import librosa
import soundfile
import torch

from spleeter.estimator import Estimator

if __name__ == '__main__':
    sr = 44100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    es = Estimator(2, './checkpoints/2stems/model').to(device)
    es.eval()

    # load wav audio
    wav, _ = librosa.load('./audio_example.mp3', mono=False, res_type='kaiser_fast',sr=sr)
    wav = torch.Tensor(wav).to(device)

    # normalize audio
    # wav_torch = wav / (wav.max() + 1e-8)

    wavs = es.separate(wav)
    for i in range(len(wavs)):
        fname = 'output/out_{}.wav'.format(i)
        print('Writing ',fname)
        soundfile.write(fname, wavs[i].cpu().detach().numpy().T, sr, "PCM_16")
        # write_wav(fname, np.asfortranarray(wavs[i].squeeze().numpy()), sr)
