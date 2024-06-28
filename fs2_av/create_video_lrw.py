import re, os, subprocess, sys
import argparse

import numpy as np

from utils.tools import *

from decord import VideoReader
from decord import cpu, gpu

import cv2
from glob import glob
from tqdm import tqdm
import pickle


def load_frames(video_file, img_size):

    # vr = VideoReader(video_file, ctx=cpu(0), width=img_size, height=img_size)
    vr = VideoReader(video_file, ctx=cpu(0))

    frames = [vr[i].asnumpy() for i in range(len(vr))]
    frames = np.asarray(frames)

    return frames


def generate(args, audio_file, result_path):

    # print("Orig audio file: ", audio_file)

    folder = audio_file.split("/")[-1].split(".")[0].split("_")[1]
    file = audio_file.split("/")[-1].split(".")[0].split("_")[-1]
    fname = folder+"_"+file
    # print("Fname: ", fname)

    video_file = os.path.join(args.video_path, folder, "test", fname+".mp4")
    # print("Video file: ", video_file)

    frames = load_frames(video_file, args.img_size)
    # print("Frames: ", frames.shape)
    
    # Save video files
    pred_video_fname = os.path.join(result_path, "pred_video", "{}".format(fname))
    # print("Pred video: ", pred_video_fname)
    generate_video(frames, audio_file, pred_video_fname)

    gt_video_fname = os.path.join(result_path, "gt_video", "{}.mp4".format(fname))
    # generate_video(frames, gt_wav_fname, gt_video_fname)
    subprocess.call('rsync %s %s' % (video_file, gt_video_fname), shell=True)
    

    return pred_video_fname

def read_file(args):

    files = np.loadtxt(args.video_path, str, delimiter=" ")
    # print(files)

    filelist = []
    for i in range(len(files)):
        filelist.append(os.path.join(args.video_data_path, files[i][0]+".mp4"))
        # filelist.append(os.path.join(args.data_path, files[i]))

    return filelist



def load_data(args):

    # if ".txt" in args.video_path:
    #     video_paths = read_file(args)
    # else:
    #     video_paths = sorted(list(glob('{}/*/*/*.mp4'.format(args.video_path))))
    audio_paths = sorted(list(glob('{}/*.wav'.format(args.audio_path))))
    print("Total audio files = ", len(audio_paths))
    # video_paths = video_paths[:2]

    
    result_path = args.result_path
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    os.makedirs(os.path.join(result_path, "pred_video"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "gt_video"), exist_ok=True)

    
    prog_bar = tqdm(audio_paths)

    
    for audio in prog_bar:

        # video = video_paths[idx]
        gen_video = generate(args, audio, result_path)

        # try:
        #     gen_video = generate(args, video, result_path)
        # except:
        #     continue
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img_size", default=160, type=int)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--video_data_path", required=False)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--result_path", default="results", type=str, help="folder path to save the results")
    
    args = parser.parse_args()

    load_data(args)