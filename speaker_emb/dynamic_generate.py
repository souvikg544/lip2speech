import synthesizer
from synthesizer import inference as sif
import numpy as np
import sys, cv2, os
from tqdm import tqdm
from shutil import copy
from glob import glob

class Generator(object):
	def __init__(self,
				synthesizer_weights='logs/'):
		super(Generator, self).__init__()

		self.synthesizer = sif.Synthesizer(synthesizer_weights, verbose=False, manual_inference=True)

	def read_window(self, window_fnames):
		window = []
		for fname in window_fnames:
			img = cv2.imread(fname)
			if img is None:
				print('Frames maybe missing in {}.' 
						' Delete the video to stop this exception!'.format(sample['folder']))

			img = cv2.resize(img, (sif.hparams.img_size, sif.hparams.img_size))
			window.append(img)

		images = np.asarray(window) / 255. # T x H x W x 3
		return images

	def vc(self, sample, outfile):
		hp = sif.hparams
		id_windows = [range(i, i + hp.T) for i in range(0, (sample['till'] // hp.T) * hp.T, 
					hp.T - hp.overlap) if (i + hp.T <= (sample['till'] // hp.T) * hp.T)]

		all_windows = [[sample['folder'].format(id) for id in window] for window in id_windows]
		last_segment = [sample['folder'].format(id) for id in range(sample['till'])][-hp.T:]
		all_windows.append(last_segment)

		try:
			ref = np.load(os.path.join(os.path.dirname(sample['folder']), 'ref.npz'))['ref'][0]
		except FileNotFoundError:
			print('Reference speaker embedding missing in {}.' 
					' Delete the video to stop this exception!'.format(sample['folder']))
			return False

		ref = np.expand_dims(ref, 0)

		for window_idx, window_fnames in enumerate(tqdm(all_windows)):
			images = self.read_window(window_fnames)

			s = self.synthesizer.synthesize_spectrograms(images, ref)[0]
			if window_idx == 0:
				mel = s
			elif window_idx == len(all_windows) - 1:
				remaining = ((sample['till'] - id_windows[-1][-1] + 1) // 5) * 16
				if remaining == 0:
					continue
				mel = np.concatenate((mel, s[:, -remaining:]), axis=1)
			else:
				mel = np.concatenate((mel, s[:, hp.mel_overlap:]), axis=1)
			
		wav = self.synthesizer.griffin_lim(mel)
		sif.audio.save_wav(wav, outfile, sr=hp.sample_rate)
		return True

if __name__ == '__main__':
	TEST_ROOT = ('../test_data/' if len(sys.argv) < 2 else sys.argv[1])
	RESULTS_ROOT = 'logs/'
	GTS_ROOT = RESULTS_ROOT + 'gts/'
	WAVS_ROOT = RESULTS_ROOT + 'wavs/'
	files_to_delete = []
	if not os.path.isdir(GTS_ROOT):
		os.mkdir(GTS_ROOT)
	else:
		files_to_delete = list(glob(GTS_ROOT + '*'))
	if not os.path.isdir(WAVS_ROOT):
		os.mkdir(WAVS_ROOT)
	else:
		files_to_delete.extend(list(glob(WAVS_ROOT + '*')))
	for f in files_to_delete: os.remove(f)

	videos = os.listdir(TEST_ROOT)
	hp = sif.hparams

	g = Generator()
	for vid in videos:
		sample = {}
		vidpath = '{}/{}/'.format(TEST_ROOT, vid)

		sample['folder'] = vidpath + '{}.jpg'

		images = glob(vidpath + '*.jpg')
		sample['till'] = (len(images) // 5) * 5
		
		outfile = WAVS_ROOT + vid + '.wav'
		success = g.vc(sample, outfile)
		if not success: continue

		copy(vidpath + 'audio.wav', GTS_ROOT + vid + '.wav')