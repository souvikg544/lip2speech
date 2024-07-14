# from tensorflow.contrib.training import HParams
from glob import glob
import os, pickle
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def _get_video_list(dataset, split, path):
    pkl_file = 'filenames_{}_{}.pkl'.format(dataset, split)
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as p:
            return pickle.load(p)
    else:
        filelist = glob(path)
        random.shuffle(filelist)

        if split == 'train':
            filelist = filelist[:int(.9995 * len(filelist))]
        else:
            filelist = filelist[int(.9995 * len(filelist)):]

        with open(pkl_file, 'wb') as p:
            pickle.dump(filelist, p, protocol=pickle.HIGHEST_PROTOCOL)

        return filelist

def _get_files_lrs2(split, path):
    fname = 'utils/filelists/{}.txt'.format(split)
    files = np.loadtxt(fname, str)

    filelist = []
    for i in range(len(files)):
        filelist.append(os.path.join(path, files[i]))

    if split=="val":
        filelist = filelist[:30]

    return filelist

def _get_all_files(split):

    
    # LRS2 train files
    filelist_lrs2 = _get_files_lrs2(split, '/ssd_scratch/cvit/souvik/mvlrs_v1/pretrain/')
    print("LRS2: ", len(filelist_lrs2))

    # LRS3 train files
    # filelist_lrs3 = _get_video_list('lrs3', split, '/ssd_scratch/cvit/sindhu/lrs3_mp4/*/*.mp4')
    # print("LRS3: ", len(filelist_lrs3))

    # LRS2 pre-train files
    # filelist_lrs2_pretrain = _get_video_list('lrs2_pretrain', split, '/ssd_scratch/cvit/sindhu/lrs2_pretrain_mp4/*/*.mp4')
    # print("LRS2 pretrain: ", len(filelist_lrs2_pretrain))

    # # LRS3 pre-train files
    # filelist_lrs3_pretrain = _get_video_list('lrs3_pretrain', split, '/ssd_scratch/cvit/sindhu/lrs3_pretrain_mp4/*/*.mp4')
    # print("LRS3 pretrain: ", len(filelist_lrs3_pretrain))
    

    # LRW pre-train files
    # filelist_lrw = _get_video_list('lrw', split, '/ssd_scratch/cvit/sindhu/lrw_trainval_mp4/*/*/*.mp4')
    # print("LRW: ", len(filelist_lrw))


    # Combine all the files
    # filelist = filelist_lrs2 + filelist_lrs3 + filelist_lrs2_pretrain + filelist_lrs3_pretrain
    filelist = filelist_lrs2
    print("Total files: ", len(filelist))

    return filelist

class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value
        
hparams = HParams(
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    #  network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value

    # For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, 
    # also consider clipping your samples to smaller chunks)
    max_mel_frames=900,
    # Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3
    #  and still getting OOM errors.
    
    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,
    
    n_fft=512,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=160,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=400,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    
    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
    
    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
    # be too big to avoid gradient explosion, 
    # not too small for fast convergence)
    normalize_for_wavenet=True,
    # whether to rescale to [0, 1] for wavenet. (better audio quality)
    clip_for_wavenet=True,
    # whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
    
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
    # levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.
    
    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.
    
    # Griffin Lim
    power=1.5,
    # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    griffin_lim_iters=60,
    # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
    ###########################################################################################################################################
    
    N=25,
    img_size=96,
    fps=25,
        
    n_gpu=1,
    batch_size=32,
    num_workers=32,
    initial_learning_rate=1e-3,
    reduced_learning_rate=None,
    nepochs=200,
    ckpt_freq=1,
    validation_interval=3,

    wav_step_size=16000,
    mel_step_size=16,
    spec_step_size=100,
    wav_step_overlap=3200,

    train_files=_get_all_files('train'),
    val_files=_get_all_files('val'),

    train_files_lrw = _get_video_list('lrw', 'train', '/ssd_scratch/cvit/sindhu/lrw_trainval_mp4/*/*/*.mp4'),
    val_files_lrw = _get_video_list('lrw', 'val', '/ssd_scratch/cvit/sindhu/lrw_trainval_mp4/*/*/*.mp4'),
)



def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)





