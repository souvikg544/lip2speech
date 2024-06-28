import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from glob import glob
import pickle
import audio_utils_librosa as au
import audio_utils as auf

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def _get_video_list(dataset, split, path):
    pkl_file = 'filenames_{}_{}.pkl'.format(dataset, split)
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as p:
            return pickle.load(p)
    else:
        filelist = glob(path)
        random.shuffle(filelist)

        if split == 'train':
            filelist = filelist[:int(.95 * len(filelist))]
        else:
            filelist = filelist[int(.95 * len(filelist)):]

        with open(pkl_file, 'wb') as p:
            pickle.dump(filelist, p, protocol=pickle.HIGHEST_PROTOCOL)

        return filelist

def _get_files_lrs2(split, path):
    fname = 'utils/filelists/{}.txt'.format(split)
    files = np.loadtxt(fname, str)

    filelist = []
    for i in range(len(files)):
        filelist.append(os.path.join(path, files[i])+".wav")

    return filelist

def _get_all_files(split):

    # LRS2 train files
    filelist_lrs2 = _get_files_lrs2(split, '/ssd_scratch/cvit/sindhu/lrs2_mp4/')
    print("LRS2: ", len(filelist_lrs2))

    # LRS3 train files
    filelist_lrs3 = _get_video_list('lrs3', split, '/ssd_scratch/cvit/sindhu/lrs3_mp4/*/*.wav')
    print("LRS3: ", len(filelist_lrs3))

    # LRS2 pre-train files
    filelist_lrs2_pretrain = _get_video_list('lrs2_pretrain', split, '/ssd_scratch/cvit/sindhu/lrs2_pretrain_mp4/*/*.wav')
    print("LRS2 pretrain: ", len(filelist_lrs2_pretrain))

    # LRS3 pre-train files
    filelist_lrs3_pretrain = _get_video_list('lrs3_pretrain', split, '/ssd_scratch/cvit/sindhu/lrs3_pretrain_mp4/*/*.wav')
    print("LRS3 pretrain: ", len(filelist_lrs3_pretrain))

    # Combine all the files
    filelist = filelist_lrs2 + filelist_lrs3 + filelist_lrs2_pretrain + filelist_lrs3_pretrain
    print("Total files: ", len(filelist))

    return filelist

mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    print("Input before pad: ", y.shape)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    print("STFT Input: ", y.shape)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)
    print("STFT Output: ", spec.shape)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def melspectrogram_librosa_filters(wav, hparams):
    D = auf._stft(auf.preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = auf._amp_to_db(auf._linear_to_mel(torch.abs(D), hparams), hparams) - hparams.ref_level_db
    
    if hparams.signal_normalization:
        return auf._normalize(S, hparams)
    return S

def melspectrogram_librosa(wav, hparams):
    D = au._stft(au.preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = au._amp_to_db(au._linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db
    
    if hparams.signal_normalization:
        return au._normalize(S, hparams)
    return S


def get_dataset_filelist(a):
    # with open(a.input_training_file, 'r', encoding='utf-8') as fi:
    #     training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
    #                       for x in fi.read().split('\n') if len(x) > 0]

    # with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
    #     validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
    #                         for x in fi.read().split('\n') if len(x) > 0]

    training_files=_get_all_files('train')
    validation_files=_get_all_files('val')

    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        # print("File: ", filename)
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
            # print("Mel: ", mel.shape)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        print("Audio: ", audio.shape)

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        print("Mel loss: ", mel_loss.shape)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)


class MelDataset_librosa_filters(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, hparams=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.hparams = hparams

    def __getitem__(self, index):
        filename = self.audio_files[index]
        # print("File: ", filename)
        if self._cache_ref_count == 0:
            # audio, sampling_rate = load_wav(filename)
            audio, sampling_rate = auf.load_wav(filename, self.hparams.sampling_rate)
            # audio = audio / MAX_WAV_VALUE
            # if not self.fine_tuning:
            #     audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            # print("Input audio: ", audio.shape, type(audio))
            mel = melspectrogram_librosa_filters(audio, self.hparams)
            # print("Mel: ", mel.shape)
        

        mel_loss = melspectrogram_librosa_filters(audio, self.hparams)
        # print("Mel loss: ", mel_loss.shape)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)



class MelDataset_librosa(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, hparams=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.hparams = hparams

    def __getitem__(self, index):
        filename = self.audio_files[index]
        # print("File: ", filename)
        if self._cache_ref_count == 0:
            audio, sampling_rate = au.load_wav(filename, self.hparams.sampling_rate)
            # audio = audio / MAX_WAV_VALUE
            # if not self.fine_tuning:
            #     audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # audio = torch.FloatTensor(audio)
        # audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.shape[0] >= self.segment_size:
                    max_audio_start = audio.shape[0] - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[audio_start:audio_start+self.segment_size]
                else:
                    audio = np.pad(audio, (0, self.segment_size - audio.shape[1]), 'constant')

            # audio = audio.squeeze(0)
            # print("Input audio: ", audio.shape, type(audio))
            mel = melspectrogram_librosa(audio, self.hparams)[:, :-1]
            # print("Mel: ", mel.shape)
        

        mel_loss = melspectrogram_librosa(audio, self.hparams)[:, :-1]
        # print("Mel loss: ", mel_loss.shape)

        mel = torch.FloatTensor(mel)
        audio = torch.FloatTensor(audio)
        mel_loss = torch.FloatTensor(mel_loss)

        return (mel, audio, filename, mel_loss)

    def __len__(self):
        return len(self.audio_files)