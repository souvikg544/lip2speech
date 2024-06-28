import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../audio/")
import hparams as hp
import audio_utils as audio
import librosa
import librosa.display

def plot_old(file, fname):

	wav = librosa.load(file, sr=16000)[0]
	stft = librosa.stft(y=wav, n_fft=hp.hparams.n_fft, hop_length=hp.hparams.hop_size, win_length=hp.hparams.win_size)
	# stft = stft[:, :420]
	print("STFT: ", stft.shape)

	# Display magnitude spectrogram
	D = np.abs(stft)
	librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),y_axis='log', x_axis='time')
	# plt.title('Power spectrogram')
	plt.colorbar(format='%+2.0f dB')
	plt.tight_layout()
	plt.show()
	plt.savefig(fname+".png")
	plt.clf()

def plot(file, fname):

	wav = librosa.load(file, sr=16000)[0][100:32100]
	melspectrogram = audio.melspectrogram(wav, hp.hparams)[:, :-1]
	print("Mels: ", melspectrogram.shape)

	# Display magnitude spectrogram
	plt.imshow(melspectrogram)
	# plt.colorbar(format='%+2.0f dB')
	plt.xticks([]), plt.yticks([])
	plt.tight_layout()
	plt.show()
	plt.savefig(fname+".png")
	plt.clf()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-g', '--gt_file', type=str, required=True, help='GT wav file')
	parser.add_argument('-p', '--pred_file', type=str, required=True, help='Predicted wav file')
	args = parser.parse_args()


	plot(args.gt_file, 'gt')
	plot(args.pred_file, 'pred_vaegan')
