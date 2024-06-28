import argparse
import audio.audio_utils as audio
import subprocess
from glob import glob
from tqdm import tqdm
import os
from os.path import dirname, join, basename, isfile
import librosa
import numpy as np
import audio.hparams as hp
# from pypesq import pesq
from pesq import pesq
from pystoi import stoi
import subprocess
from scipy.io import wavfile
import soundfile as sf

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# from fastprogress.fastprogress import progress_bar


def compute_metrics(args):

	pred_files = glob(os.path.join("{}/*.wav".format(args.pred_files)))
	print("Total files = ", len(pred_files))

	total_stoi = []
	total_pesq = []
	total_estoi = []

	prog_bar = tqdm(range(len(pred_files)))
	for idx in prog_bar:
		pred_wav_filename = pred_files[idx]

		fname = pred_wav_filename.split('/')[-1]
		gt_wav_filename = os.path.join(args.gt_files, fname)
		# print(gt_wav_filename)
			
		# pred_wav = librosa.load(pred_wav_filename, sr=16000)[0]
		# gt_wav = librosa.load(gt_wav_filename, sr=16000)[0]
		fs = 16000

		rate, pred_wav = wavfile.read(pred_wav_filename)
		rate, gt_wav = wavfile.read(gt_wav_filename)
		rate = 16000

		# gt_wav, fs = sf.read(gt_wav_filename)
		# gt_wav = gt_wav[1600:]
		# pred_wav, fs = sf.read(pred_wav_filename)
		# pred_wav = pred_wav[1600:]
		# print(fs)

		if gt_wav.shape[0] > pred_wav.shape[0]:
			gt_wav = gt_wav[:pred_wav.shape[0]]
		elif pred_wav.shape[0] > gt_wav.shape[0]:
			pred_wav = pred_wav[:gt_wav.shape[0]]
		
		# pesq_metric = pesq(gt_wav, pred_wav, 16000)
		pesq_metric = pesq(rate, gt_wav, pred_wav, 'nb')
		if np.isnan(pesq_metric) or pesq_metric == -1:
			continue

		total_pesq.append(pesq_metric)
		total_stoi.append(stoi(gt_wav, pred_wav, fs, extended=False))
		total_estoi.append(stoi(gt_wav, pred_wav, fs, extended=True))

		prog_bar.set_description('PESQ: %.3f, STOI: %.4f, ESTOI: %.4f' % 
										   ((sum(total_pesq) / len(total_pesq)),
										   (sum(total_stoi) / len(total_stoi)),
										   (sum(total_estoi) / len(total_estoi))))
		prog_bar.refresh()
			
	print('Mean PESQ: {}'.format(sum(total_pesq) / len(total_pesq)))
	print('Mean STOI: {}'.format(sum(total_stoi) / len(total_estoi)))
	print('Mean ESTOI: {}'.format(sum(total_estoi) / len(total_estoi)))

	print("------------------------------------------------")



if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-gt', '--gt_files', type=str, required=True, \
						help='Folder of GT wav files')
	parser.add_argument('-p', '--pred_files', type=str,  required=True, \
						help= 'Folder of predicted wav file')

	args = parser.parse_args()

	compute_metrics(args)