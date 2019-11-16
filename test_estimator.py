import torch
import torchaudio
from librosa.core import load
from librosa.output import write_wav

from spleeter.estimator import Estimator

es = Estimator(2, './checkpoints/2stems/model')

# load wav audio
wav, sr = torchaudio.load_wav('./audio_example.mp3')

# normalize audio 
wav_torch = wav / (wav.max() + 1e-8)

wavs = es.separate(wav_torch)
for i in range(len(wavs)):
    fname = 'output/out_{}.wav'.format(i)
    print('Writing ',fname)
    write_wav(fname, wavs[i].squeeze().numpy(), sr)
