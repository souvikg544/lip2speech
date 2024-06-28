import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D

from decord import VideoReader
from decord import cpu, gpu

import random

import librosa
import audio as Audio
import audio.hparams as hp
import audio.audio_utils as au

from g2p_en import G2p
from string import punctuation
import re

from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import torch

mel_basis = {}
hann_window = {}

class Dataset(Dataset):
	def __init__(
		self, preprocess_config, train_config, sort=False, drop_last=False, test=False
	):
		self.batch_size = train_config["optimizer"]["batch_size"]
		self.img_size = preprocess_config["preprocessing"]["img_size"]
		self.scale_factor = preprocess_config["preprocessing"]["upsample_scale_mel"]
		self.sort = sort
		self.drop_last = drop_last
		self.test = test
		self.files = hp.hparams.train_files if not self.test else hp.hparams.val_files
		self.preprocess_config = preprocess_config
		self.g2p = G2p()
		self.max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]

		self.n_fft = preprocess_config["preprocessing"]["stft"]["filter_length"]
		self.hop_size = preprocess_config["preprocessing"]["stft"]["hop_length"]
		self.win_size = preprocess_config["preprocessing"]["stft"]["win_length"]
		self.num_mels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
		self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
		self.fmin = preprocess_config["preprocessing"]["mel"]["mel_fmin"]
		self.fmax = preprocess_config["preprocessing"]["mel"]["mel_fmax"]

		# self.STFT = Audio.stft.TacotronSTFT(
		# 	preprocess_config["preprocessing"]["stft"]["filter_length"],
		# 	preprocess_config["preprocessing"]["stft"]["hop_length"],
		# 	preprocess_config["preprocessing"]["stft"]["win_length"],
		# 	preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
		# 	preprocess_config["preprocessing"]["audio"]["sampling_rate"],
		# 	preprocess_config["preprocessing"]["mel"]["mel_fmin"],
		# 	preprocess_config["preprocessing"]["mel"]["mel_fmax"],
		# )

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		
		while(1):
			idx = random.randint(0, len(self.files) - 1)
			video_file = self.files[idx]
			if ".mp4" not in video_file:
				video_file = video_file+".mp4"
			# print("Video: ", video_file)

			# raw_text = None
			# if self.test:
			#     text_file = video_file.replace(".mp4", ".txt")
			#     with open(text_file) as f:
			#         raw_text = f.read().splitlines()
			#     raw_text = raw_text[0].split(":  ")[1]
			try:
				se_path = video_file.replace(".mp4", ".npz")
				speaker_emb = np.load(se_path)['ref'][0]
			except:
				continue
			# print("Speaker emb: ", speaker_emb.shape)

			try:
				phoneme_file = video_file.replace(".mp4", "_phonemes.npy")
				phone = np.load(phoneme_file)            
			except:
				continue

			phoneme_file = video_file.replace(".mp4", "_phonemes.npy")
			phone = np.load(phoneme_file)            
			# print("Ph: ", phone)


			visual_emb, frames = self.load_frames(video_file, test=self.test)
			if visual_emb is None or visual_emb.shape[0]>1000:
				continue

			wav_file = video_file.replace(".mp4", ".wav")
			wav, _ = librosa.load(wav_file, sr=16000)
			# wav = wav / max(abs(wav)) * self.max_wav_value
			wav = wav / self.max_wav_value
			wav = normalize(wav) * 0.95
			# mel_spectrogram = au.melspectrogram(wav, hp.hparams).T[:-1]
			# mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, self.STFT)
			# mel_spectrogram = mel_spectrogram.T[:-1]
			audio = torch.FloatTensor(wav)
			audio = audio.unsqueeze(0)
			mel_spectrogram = self.mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, \
										self.win_size, self.fmin, self.fmax, center=False)
			mel_spectrogram = mel_spectrogram.squeeze(0).cpu().numpy().T
			# print("Mels: ", mel_spectrogram.shape, "Mels vocoder: ", mel.shape)
			if mel_spectrogram.shape[0] > visual_emb.shape[0]*self.scale_factor:
				mel_spectrogram = mel_spectrogram[:visual_emb.shape[0]*self.scale_factor]
			# print("Mels: ", mel_spectrogram.shape)
 
			
			
			sample = {
				"text": phone,
				"mel": mel_spectrogram,
				"visual_emb": visual_emb,
				"speaker_emb": speaker_emb,
				"frames": frames
			}
			# print("Sample: ", sample)
			

			return sample

	def mel_spectrogram(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):

		if torch.min(y) < -1.:
			print('min value is ', torch.min(y))
		if torch.max(y) > 1.:
			print('max value is ', torch.max(y))

		
		global mel_basis, hann_window
		if fmax not in mel_basis:
			mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
			mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
			hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

		y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
		y = y.squeeze(1)

		spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
						  center=center, pad_mode='reflect', normalized=False, onesided=True)

		spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

		spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
		spec = self.spectral_normalize_torch(spec)

		return spec

	def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
		return torch.log(torch.clamp(x, min=clip_val) * C)

	def spectral_normalize_torch(self, magnitudes):
		output = self.dynamic_range_compression_torch(magnitudes)
		return output

	def load_frames(self, video_file, test):

		try:
			visual_emb_file = video_file.replace(".mp4", "_vtp.npy")
			visual_emb = np.load(visual_emb_file)
		except:
			return None, None
		# print("Visual emb: ", visual_emb.shape)                     # n_framesx512

		frames=None
		if test:
			try:
				vr = VideoReader(video_file, ctx=cpu(0), width=self.img_size, height=self.img_size)
			except:
				return None, None

			frames = [vr[i].asnumpy() for i in range(len(vr))]
			frames = np.asarray(frames)

		return visual_emb, frames

	def read_lexicon(self, lex_path):
		lexicon = {}
		with open(lex_path) as f:
			for line in f:
				temp = re.split(r"\s+", line.strip("\n"))
				word = temp[0]
				phones = temp[1:]
				if word.lower() not in lexicon:
					lexicon[word.lower()] = phones
		return lexicon


	def preprocess_english(self, text, preprocess_config):
		text = text.rstrip(punctuation)
		lexicon = self.read_lexicon(preprocess_config["path"]["lexicon_path"])

		
		phones = []
		words = re.split(r"([,;.\-\?\!\s+])", text)
		for w in words:
			if w.lower() in lexicon:
				phones += lexicon[w.lower()]
			else:
				phones += list(filter(lambda p: p != " ", self.g2p(w)))
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

	def process_meta(self, filename):
		with open(
			os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
		) as f:
			name = []
			speaker = []
			text = []
			raw_text = []
			for line in f.readlines():
				n, s, t, r = line.strip("\n").split("|")
				name.append(n)
				speaker.append(s)
				text.append(t)
				raw_text.append(r)
			return name, speaker, text, raw_text

	def reprocess(self, data, idxs):
		texts = [data[idx]["text"] for idx in idxs]
		mels = [data[idx]["mel"] for idx in idxs]
		visual_emb = [data[idx]["visual_emb"] for idx in idxs]
		speaker_emb = [data[idx]["speaker_emb"] for idx in idxs]
		frames = [data[idx]["frames"] for idx in idxs]

		text_lens = np.array([text.shape[0] for text in texts])
		mel_lens = np.array([mel.shape[0] for mel in mels])
		video_lens = np.array([emb.shape[0] for emb in visual_emb])
		mel_lens_video = np.array([int(vid_len*self.scale_factor) for vid_len in video_lens])

		texts = pad_1D(texts)
		mels = pad_2D(mels)
		visual_emb = pad_2D(visual_emb)
		speaker_emb = np.array(speaker_emb)
		frames = np.array(frames)

		return (
			texts,
			text_lens,
			max(text_lens),
			visual_emb,
			video_lens,
			max(video_lens),
			mel_lens_video,
			max(mel_lens_video),
			speaker_emb,
			mels,
			mel_lens,
			max(mel_lens),
			frames
		)

	def collate_fn(self, data):
		data_size = len(data)
		# print("data size: ", data_size)

		if self.sort:
			len_arr = np.array([d["text"].shape[0] for d in data])
			idx_arr = np.argsort(-len_arr)
		else:
			idx_arr = np.arange(data_size)

		tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
		idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
		idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
		if not self.drop_last and len(tail) > 0:
			idx_arr += [tail.tolist()]

		# print("IDX ARR:", len(idx_arr))
		output = list()
		for idx in idx_arr:
			output.append(self.reprocess(data, idx))

		# print("Output: ", output)
		return output