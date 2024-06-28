import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
from glob import glob



def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    # print("Out dir: ", out_dir)
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    for speaker in tqdm(os.listdir(in_dir)):
        for file_name in glob(os.path.join(in_dir, speaker, "*.wav")):
            # print("File: ", file_name)
            if ".wav" not in file_name:
                continue
            
            base_name = os.path.basename(file_name).split(".")[0]
            # print("Basename: ", base_name)
            
            text_path = os.path.join(
                in_dir, speaker, "{}.txt".format(base_name)
            )
            
            wav_path = os.path.join(
                in_dir, speaker, "{}.wav".format(base_name)
            )
            
            with open(text_path) as f:
                text = f.readline().strip("\n")
            text = _clean_text(text, cleaners)
            # print("Cleaned text: ", text)

            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sr=sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wav_fname = os.path.join(out_dir, speaker, "{}.wav".format(base_name))

            # print("Output file: ", wav_fname)

            wavfile.write(
                wav_fname,
                sampling_rate,
                wav.astype(np.int16),
            )
            
            with open(os.path.join(out_dir, speaker, "{}.lab".format(base_name)),"w") as f1:
                f1.write(text)

            # exit(0)