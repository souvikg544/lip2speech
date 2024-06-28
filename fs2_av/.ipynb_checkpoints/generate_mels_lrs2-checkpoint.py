import re, os, subprocess, sys
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model_mel, get_vocoder
from utils.tools import *
from dataset import TextDataset
from text import text_to_sequence

from decord import VideoReader
from decord import cpu, gpu

import cv2
from glob import glob
from tqdm import tqdm
import pickle

import traceback
import executor
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    # print("Raw Text Sequence: {}".format(text))
    # print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)



def load_frames(visual_emb_file, video_file, img_size):

    visual_emb = np.load(visual_emb_file)
    # print("Visual emb inside: ", visual_emb.shape)                     # n_framesx512

    vr = VideoReader(video_file, ctx=cpu(0), width=img_size, height=img_size)

    frames = [vr[i].asnumpy() for i in range(len(vr))]
    frames = np.asarray(frames)

    return visual_emb, frames


def get_predictions(args, video_file, model, configs, text_dict, result_path):

    preprocess_config, model_config, train_config = configs
    
    speakers = np.array([0])

    if text_dict is not None:
        if ".txt" in args.lr_model_pred:
            ## LRS3
            # pred_key = video_file.split("/")[-2] + "/" + video_file.split("/")[-1]
            # raw_texts = text_dict[pred_key]
            ## TIMIT
            # pred_key = video_file.split("/")[-3] + "_" + video_file.split("/")[-1].split("_crop")[0]
            # raw_texts = text_dict[pred_key]                
            ## Deep LR
            if "deep_lip_reading" in args.lr_model_pred:
                pred_key = video_file.split("/")[-2] + "/" + video_file.split("/")[-1].split(".")[0] + ".mp4"
                raw_texts = text_dict[pred_key]
            ## AVHubert LRS2           
            elif "avhubert" in args.lr_model_pred:
                pred_key = "/content/LRS2_testset/" + video_file.split("/")[-2] + "/" + video_file.split("/")[-1].split(".")[0] + "_MOUTH.mp4"
                raw_texts = text_dict[pred_key] 
            # print("Key: ", pred_key, "Raw text: ", raw_texts)                      
        elif ".pkl" in args.lr_model_pred:
            # print(text_dict)
            # LRS2
            # pred_key = "lrs2/vid/" + video_file.split("/")[-2] + "/" + video_file.split("/")[-1].split(".")[0] + ".npy"
            # raw_texts = text_dict[pred_key]['preds'][0]
            # LRW
            pred_key = video_file.split("/")[-3] + "/" + video_file.split("/")[-2] + "/" + video_file.split("/")[-1].split(".")[0] + ".npy"
            raw_texts = text_dict[pred_key]  
        ids = raw_texts
    else:
        text_file = video_file.replace("_crop.mp4", ".txt")
        with open(text_file) as f:
            raw_texts = f.read().splitlines()
        # raw_texts = raw_texts[0].split(":  ")[1]
        raw_texts = raw_texts[0]
        ids = raw_texts
    # print("Raw text: ", raw_texts)
    texts = np.array([preprocess_english(raw_texts, preprocess_config)])
    text_lens = np.array([len(texts[0])])

    speaker_emb_file = video_file.replace(".mp4", ".npz")
    speaker_emb = np.array([np.load(speaker_emb_file, allow_pickle=True)['ref'][0]])

    visual_emb_file = video_file.replace(".mp4", "_vtp.npy")
    visual_emb, frames = load_frames(visual_emb_file, video_file, args.img_size)
    visual_emb = np.array([visual_emb])

    video_lens = np.array([len(visual_emb[0])])
    scale_factor = preprocess_config["preprocessing"]["upsample_scale_mel"]
    mel_lens_video = np.array([int(video_lens*scale_factor)])

    batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens), \
        visual_emb, video_lens, max(video_lens), mel_lens_video, max(mel_lens_video), speaker_emb, frames)

    # batch = (texts, text_lens, max(text_lens), \
    #     visual_emb, video_lens, max(video_lens), mel_lens_video, max(mel_lens_video), speaker_emb, frames)

    batch = to_device(batch, device, train=False)

    with torch.no_grad():
        output = model(*(batch[2:]))
        # output = model(*(batch))

    pred_wav, pred_mel = synth_sample_generation(batch, output, model_config, preprocess_config)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    fname = video_file.split("/")[-2] + "_" + video_file.split("/")[-1].split(".")[0]

    # Save audio files
    pred_wav_fname = os.path.join(result_path, "pred_audio", "{}.wav".format(fname))
    wavfile.write(pred_wav_fname, sampling_rate, pred_wav)

    gt_wav_fname = os.path.join(result_path, "gt_audio", "{}.wav".format(fname))
    subprocess.call('rsync %s %s' % (video_file.replace(".mp4", ".wav"), gt_wav_fname), shell=True)

    
    # Save video files
    pred_video_fname = os.path.join(result_path, "pred_video", "{}".format(fname))
    generate_video_inference(frames, pred_wav_fname, pred_video_fname)

    gt_video_fname = os.path.join(result_path, "gt_video", "{}.mp4".format(fname))
    # generate_video(frames, gt_wav_fname, gt_video_fname)
    subprocess.call('rsync %s %s' % (video_file, gt_video_fname), shell=True)
    

    if args.save_mels:
        pred_mel_fname = os.path.join(result_path, "pred_mels", "{}.npy".format(fname))
        np.save(pred_mel_fname, pred_mel.cpu().numpy())

    return pred_wav_fname

def read_file(args):

    files = np.loadtxt(args.video_root, str, delimiter=" ")
    # print(files)

    filelist = []
    for i in range(len(files)):
        filelist.append(os.path.join(args.data_path, files[i][0]+".mp4"))
        # filelist.append(os.path.join(args.data_path, files[i]))

    return filelist

def read_file_timit(args):

    files_sp1 = np.loadtxt("utils/filelists/sp1_test.txt", str, delimiter=" ")
    files_sp2 = np.loadtxt("utils/filelists/sp2_test.txt", str, delimiter=" ")
    # print(files)

    filelist = []
    for i in range(len(files_sp1)):
        filelist.append(os.path.join(args.data_path, "speaker1/straightcam", files_sp1[i]+"_crop.mp4"))
    for j in range(len(files_sp2)):
        filelist.append(os.path.join(args.data_path, "speaker2/straightcam", files_sp2[j]+"_crop.mp4"))

    return filelist

def mp_handler(args, file, model, configs, text_dict, result_path):

    try:
        pred_wav_filename = get_predictions(args, file, model, configs, text_dict, result_path)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()

def load_data(args, model, configs):

    if ".txt" in args.video_root:
        video_paths = read_file(args)
        # video_paths = read_file_timit(args)
    else:
        video_paths = sorted(list(glob('{}/*/*.mp4'.format(args.video_root))))
    print("Total test videos = ", len(video_paths))
    # video_paths = video_paths[:2]

    if args.lr_model_pred is not None:
        if ".txt" in args.lr_model_pred:
            raw_data = np.loadtxt(args.lr_model_pred, str, delimiter=':')
            text_dict = {}
            for i in range(len(raw_data)):
                key = raw_data[i][0]
                value = raw_data[i][1]
                text_dict[key] = value
        elif ".pkl" in args.lr_model_pred:
            with open(args.lr_model_pred, 'rb') as p:
                text_dict = pickle.load(p)
    else:
        text_dict = None

    # print("Text dict: ", text_dict)
    result_path = args.result_path
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    os.makedirs(os.path.join(result_path, "pred_audio"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "gt_audio"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "pred_video"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "gt_video"), exist_ok=True)

    if args.save_mels:
        os.makedirs(os.path.join(result_path, "pred_mels"), exist_ok=True)

    prog_bar = tqdm(range(len(video_paths)))

    '''
    for idx in prog_bar:

        video = video_paths[idx]
        # pred_wav_filename = get_predictions(args, video, model, configs, text_dict, result_path)

        try:
            pred_wav_filename = get_predictions(args, video, model, configs, text_dict, result_path)
        except:
            continue
    '''

    jobs = [file for file in video_paths]
    p = ThreadPoolExecutor(4)
    futures = [p.submit(mp_handler, args, j, model, configs, text_dict, result_path) for j in jobs]
    res = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    
    parser.add_argument("-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str, required=True, help="path to model.yaml")
    parser.add_argument("-t", "--train_config", type=str, required=True, help="path to train.yaml")
    
    parser.add_argument("--img_size", default=160, type=int)
    parser.add_argument("--video_root", required=True, help="path of the folder containing input videos")
    parser.add_argument("--data_path", required=False)
    parser.add_argument("--lr_model_pred", default=None, help="predictions from lip reading model")
    parser.add_argument("--result_path", default="results", type=str, help="folder path to save the results")
    parser.add_argument("--save_mels", default=False, type=str)
    
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model_mel(args, configs, device, train=False)
    model = model.cuda()

    load_data(args, model, configs)