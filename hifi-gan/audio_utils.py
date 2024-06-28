import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import torch
import torchaudio

_mel_basis = None

def load_wav(path, sr=16000):
    return librosa.core.load(path, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        # sig =  signal.lfilter([1, -k], [1], wav)
        sig = torchaudio.functional.lfilter(wav, torch.tensor([1, 0]).to(wav.device), torch.tensor([1, -k]).to(wav. device))
        # print("preemphasize signal: ", sig.shape, type(sig))
        return sig
    return wav

def _stft(y, hparams):
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        # return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size)
        y = y.squeeze(1)
        spec = torch.stft(y, hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size, return_complex=True)[:, :, :-1]
        # print("Torch stft: ", spec.shape)
        return spec

def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sampling_rate)
    return hop_size

def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams, spectogram.device)
        # _mel_basis = _mel_basis.reshape(1,_mel_basis.shape[0], _mel_basis.shape[1])
        # _mel_basis = _mel_basis.unsqueeze(0)
    # print("Mel basis: ", _mel_basis.shape)
    # print("Spec: ", spectogram.shape)
    # print("Melbasis dev: ", _mel_basis.device)
    # print("Spec device: ", spectogram.device)
    return torch.matmul(_mel_basis.to(spectogram.device), spectogram)

def _build_mel_basis(hparams, device):
    assert hparams.fmax <= hparams.sampling_rate // 2
    return torch.from_numpy(librosa.filters.mel(hparams.sampling_rate, hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax))

def _amp_to_db(x, hparams):
    min_level = torch.exp(hparams.min_level_db / 20 * torch.log(torch.tensor(10))).to(x.device)
    return 20 * torch.log10(torch.maximum(min_level, x))

def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return torch.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return torch.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))
