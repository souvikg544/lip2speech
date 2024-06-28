import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model_all, get_vocoder
from utils.tools import *
from model import FastSpeech2LossAll
from dataset_all_librosa_lrw import Dataset

from scipy.io import wavfile

import cv2, subprocess
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(preprocess_config, train_config, sort=False, drop_last=False, test=True)
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2LossAll(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    for batchs in tqdm(loader):
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                try:
                    output = model(*(batch))
                except:
                    continue

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}".format(step),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_reconstructed".format(step),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_synthesized".format(step),
        )

        result_wav_path = os.path.join(train_config["path"]["result_path"], "val", "wavs")
        if not os.path.exists(result_wav_path):
            os.makedirs(result_wav_path)
        gt_wav_fpath = os.path.join(result_wav_path, "{}_gt.wav".format(step))
        wavfile.write(gt_wav_fpath, sampling_rate, wav_reconstruction)
        pred_wav_fpath = os.path.join(result_wav_path, "{}_pred.wav".format(step))
        wavfile.write(pred_wav_fpath, sampling_rate, wav_prediction)
        print("Saved the reconstructions at: ", result_wav_path)

        result_video_path = os.path.join(train_config["path"]["result_path"], "val", "videos")
        if not os.path.exists(result_video_path):
            os.makedirs(result_video_path)
        video_fpath = os.path.join(result_video_path, "{}".format(step))
        frames = batch[-1][0]
        save_video(gt_wav_fpath, pred_wav_fpath, frames, video_fpath)
        print("Saved the videos at: ", result_video_path)

    return message



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
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
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model_all(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)