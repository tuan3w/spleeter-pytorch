# spleeter-pytorch
Spleeter implementation in pytorch


## Usage

```python
from spleeter.estimator import Estimator
import torchaudio
from librosa.core import load
from librosa.output import write_wav
import torch

es = Estimator(2, './checkpoints/model')

wav, sr = torchaudio.load_wav('./audio_example.mp3')
# import pdb; pdb.set_trace()
wav_torch = wav / (wav.max() + 1e-8)
print(wav_torch.min(), )
wavs = es.separate(wav_torch)
for i in range(len(wavs)):
    fname = 'out_{}.wav'.format(i)
    print('Writing ',fname)
    write_wav(fname, wavs[i].squeeze().numpy(), sr)
```

Note: There are some bugs in code, I can't figure out the reason it doesnt work well as origin repo. If someone can figure out the problem, please send me a merge request. Thanks.
