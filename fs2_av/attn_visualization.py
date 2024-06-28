import re, os, subprocess, sys
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model_all, get_vocoder
from utils.tools import to_device, synth_one_sample_inference
from dataset import TextDataset
from text import text_to_sequence

from decord import VideoReader
from decord import cpu, gpu

import cv2
import pickle
import matplotlib.pyplot as plt

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

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)



def synthesize(model, step, configs, vocoder, batchs, result_path, result_fname):
    preprocess_config, model_config, train_config = configs

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for batch in batchs:
        batch = to_device(batch, device, train=False)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch),
            )
            synth_one_sample_inference(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                # train_config["path"]["result_path"],
                result_path,
                result_fname
            )
            attn_wts = output[-1].detach().cpu().numpy()[0]
            print("Attn wts: ", attn_wts.shape)
            np.save("attn.npy", attn_wts)

            attn_wts = attn_wts.T
            plt.imshow(attn_wts, origin='lower')
            plt.show()
            plt.savefig("attn.png")



def load_frames_vtp(visual_emb_file, video_file, img_size):

    visual_emb = np.load(visual_emb_file)
    # print("Visual emb inside: ", visual_emb.shape)                     # n_framesx512

    vr = VideoReader(video_file, ctx=cpu(0), width=img_size, height=img_size)

    frames = [vr[i].asnumpy() for i in range(len(vr))]
    frames = np.asarray(frames)

    return visual_emb, frames

def load_frames(video_file, img_size):

    vr = VideoReader(video_file, ctx=cpu(0), width=img_size, height=img_size)

    frames = [vr[i].asnumpy() for i in range(len(vr))]
    frames = np.asarray(frames)

    return frames

def get_speaker_emb(args):

    sys.path.append('../../avse/speaker_emb/')
    import encoder
    from encoder import inference as eif
    
    encoder_weights = '../../avse/speaker_emb/encoder/saved_models/pretrained.pt'
    eif.load_model(encoder_weights)
    secs = 1
    k = 1

    wav = encoder.audio.preprocess_wav(args.ref_speaker)
    # print("Wav: ", wav.shape)
    if len(wav) < secs * encoder.audio.sampling_rate: 
        return

    # wav_segment = wav[:16000,]
    wav_segment = wav
    # print("Wav segment: ", wav_segment.shape)

    embeddings = np.asarray(eif.embed_utterance(wav_segment))
    print("Speaker emb: ", embeddings.shape)

    return embeddings

def get_visual_emb(args):

    ckpt_path = '/ssd_scratch/cvit/sindhu/feature_extractor.pth'
    video_path = args.video_file
    feature_path = os.path.join(args.result_path, args.result_fname+'_vtp.npy')

    '''
    sys.path.append('/home/sindhu/Research/l2s_tts/')
    import vtp
    from vtp import extract_emb_inf as eif

    visual_emb = eif.save_visual_emb(video_path, feature_path, ckpt_path)
    '''

    subprocess.call('python ../vtp/extract_emb_inf.py --videos_root=%s --feats_root=%s --ckpt_path=%s' % \
                    (video_path, feature_path, ckpt_path), shell=True)
    visual_emb = np.load(feature_path)

    return visual_emb


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default="single",
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="raw text file to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    
    parser.add_argument("--img_size", default=160, type=int)
    parser.add_argument("--video_file", required=True, help="path of the video file")
    # parser.add_argument("--ref_speaker", required=True, help="audio file of the reference speaker (for voice info)")
    parser.add_argument("--lr_model_pred", default=None, help="predictions from lip reading model")
    parser.add_argument("--visual_emb_file", required=True, help="path of the dumped visual embedding")
    parser.add_argument("--speaker_emb_file", required=True, help="path of the speaker embedding file (for voice info)")
    parser.add_argument("--result_path", default="results", type=str, help="folder path to save the results")
    parser.add_argument("--result_fname", default="result", type=str, help="name of the result file(s)")
    
    
    args = parser.parse_args()

    # Check source texts
    # if args.mode == "batch":
    #     assert args.source is not None and args.text is None
    # if args.mode == "single":
    #     assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model_all(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    # CURRENTLY NOT SUPPORTED
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        # raw_texts = np.loadtxt(args.text_file, str, delimiter=':  ')[0][1]
        if args.text_file is not None:
            with open(args.text_file) as f:
                raw_texts = f.read().splitlines()
            raw_texts = raw_texts[0].split(":  ")[1]
        elif args.lr_model_pred is not None:
            if "lrs3" in args.lr_model_pred:
                raw_data = np.loadtxt(args.lr_model_pred, str, delimiter=',')
                text_dict = {}
                for i in range(len(raw_data)):
                    key = raw_data[i][0]
                    value = raw_data[i][1]
                    text_dict[key] = value
                pred_key = args.video_file.split("/")[-2] + "/" + args.video_file.split("/")[-1]
                raw_texts = text_dict[pred_key]
            elif "lrs2" in args.lr_model_pred:
                with open(args.lr_model_pred, 'rb') as p:
                    text_dict = pickle.load(p)

                pred_key = "lrs2/vid/" + args.video_file.split("/")[-2] + "/" + args.video_file.split("/")[-1].split(".")[0] + ".npy"
                raw_texts = text_dict[pred_key]['preds'][0]
                print("text: ", raw_texts)
                ids = raw_texts
            

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(raw_texts, preprocess_config)])
        print("Texts: ", texts.shape)
        text_lens = np.array([len(texts[0])])

        speaker_emb = np.array([np.load(args.speaker_emb_file)['ref'][0]])
        visual_emb, frames = load_frames_vtp(args.visual_emb_file, args.video_file, args.img_size)

        # speaker_emb = get_speaker_emb(args)
        # speaker_emb = np.array([speaker_emb])

        # frames = load_frames(args.video_file, args.img_size)
        # visual_emb = get_visual_emb(args)
        visual_emb = np.array([visual_emb])
        print("Visual emb: ", visual_emb.shape)                     

        video_lens = np.array([len(visual_emb[0])])
        scale_factor = preprocess_config["preprocessing"]["upsample_scale_mel"]
        mel_lens_video = np.array([int(video_lens*scale_factor)])

        batchs = [(texts, text_lens, max(text_lens), visual_emb, video_lens, max(video_lens), \
            mel_lens_video, max(mel_lens_video), speaker_emb, frames)]


    synthesize(model, args.restore_step, configs, vocoder, batchs, args.result_path, args.result_fname)
