import sys, torch

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

# torch.backends.cudnn.benchmark = False

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import listdir, path
import numpy as np
import argparse, os, traceback
from tqdm import tqdm
from glob import glob
import encoder, subprocess
from encoder import inference as eif
from synthesizer import audio as sa
from synthesizer import hparams as hp
import subprocess

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

encoder_weights = 'encoder/saved_models/pretrained.pt'
eif.load_model(encoder_weights)
secs = 1
k = 1

command = 'ffmpeg -i {} {}'

def process_video_file(afile, args):
	# print("Audio file: ", afile)
	command_to_run = command.format(afile, afile.replace('.mp4', '.wav'))
	subprocess.call(command_to_run, shell=True)
	# wav = encoder.audio.preprocess_wav(afile)
	# # print("Wav: ", wav.shape)
	# if len(wav) < secs * encoder.audio.sampling_rate: return

	# indices = np.random.choice(len(wav) - encoder.audio.sampling_rate * secs, k)
	# wavs = [wav[idx : idx + encoder.audio.sampling_rate * secs] for idx in indices]
	# embeddings = np.asarray([eif.embed_utterance(wav) for wav in wavs])
	# # print("Emb: ", embeddings.shape)
	# # print("Afile: ", afile.replace('.wav', '.npz'))
	# np.savez_compressed(afile.replace('.', '.npz'), ref=embeddings)

	# wav = sa.load_wav(afile, sr=hp.hparams.sample_rate)
	# lspec = sa.linearspectrogram(wav, hp.hparams)
	# melspec = sa.melspectrogram(wav, hp.hparams)

	# np.savez_compressed(afile.replace('audio.wav', 'mels.npz'), lspec=lspec, mel=melspec)

def mp_handler(job):
	vfile, args = job
	try:
		process_video_file(vfile, args)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def dump(args):
	print('Started processing for with {} CPU cores'.format(args.num_workers))

	filelist = glob(path.join(args.final_data_root, '*',  '*.mp4'))
	print("Total files: ", len(filelist))

	jobs = [(vfile, args) for vfile in filelist]
	p = ThreadPoolExecutor(args.num_workers)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', help='Number of workers to run in parallel', default=16, type=int)

parser.add_argument("--final_data_root", help="Folder where preprocessed files will reside", 
					default='/ssd_scratch/cvit/souvik/pretrain/')

args = parser.parse_args()

if __name__ == '__main__':
	dump(args)
