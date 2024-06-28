import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, FastSpeech2Mel, FastSpeech2All, FastSpeech2AllFace, ScheduledOptim

import audio as Audio
import audio.hparams as hp
import audio.audio_utils as au
from scipy.io import wavfile

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_mel(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2Mel(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_model_all(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2All(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_model_all_face(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2AllFace(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config_v1.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            # ckpt = torch.load("hifigan/generator_universal.pth.tar")
            ckpt = torch.load("/scratch/sindhu/vocoder_hifigan/g_00400000")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

def vocoder_infer_mel(mels, model_config, preprocess_config, lengths=None):
    
    mels = mels.squeeze(0).detach().cpu().numpy()
    wavs = au.inv_mel_spectrogram(mels, hp.hparams)
    
    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

def vocoder_infer_mel_all(mels, model_config, preprocess_config, lengths=None):
    '''
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)    
    '''
    STFT = Audio.stft.TacotronSTFT(
                preprocess_config["preprocessing"]["stft"]["filter_length"],
                preprocess_config["preprocessing"]["stft"]["hop_length"],
                preprocess_config["preprocessing"]["stft"]["win_length"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                preprocess_config["preprocessing"]["mel"]["mel_fmin"],
                preprocess_config["preprocessing"]["mel"]["mel_fmax"],
            )

    # mels = mels.squeeze(0).detach().cpu().numpy()
    # print("Mels: ", mels.shape)
    # inv_stft = Audio.tools.InvSTFT(preprocess_config)
    wavs = Audio.tools.inv_mel_spec(mels, STFT)
    # wavs = STFT.inv_mel_spec(mels)
    # wavfile.write("trial.wav", 16000, wavs)
    # wavs = np.expand_dims(wavs, axis=0).astype("int16")
    # print("Wavs: ", wavs.shape)
    # wavs = (wavs.cpu().numpy()).astype("int16")
    # wavs = (wavs * preprocess_config["preprocessing"]["audio"]["max_wav_value"]).astype("int16")
    # wavs = [wav/max(0.01, np.max(np.abs(wav))) for wav in wavs]
    # wavs = [wav for wav in wavs]
    
    
    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

def vocoder_infer_hifigan(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")

    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs