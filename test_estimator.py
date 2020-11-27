import torchaudio
import soundfile as sf

from spleeter.estimator import Estimator
import os


es = Estimator(2, './checkpoints/2stems/model')


def main(original_audio='./audio_example.mp3', out_dir='./output'):
    # load wav audio
    wav, sr = torchaudio.load(original_audio)

    # normalize audio
    wav_torch = wav / (wav.max() + 1e-8)

    wavs = es.separate(wav_torch)
    for i in range(len(wavs)):
        fname = os.path.join(out_dir, f'out_{i}.wav')
        print('Writing:', fname)
        new_wav = wavs[i].squeeze()
        new_wav = new_wav.permute(1, 0)
        new_wav = new_wav.numpy()
        sf.write(fname, new_wav, sr)


if __name__ == '__main__':
    main()
