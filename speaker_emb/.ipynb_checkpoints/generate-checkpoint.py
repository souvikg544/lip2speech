import synthesizer
from synthesizer import inference as sif
import numpy as np
import sys, cv2, os
from tqdm import tqdm

class Generator(object):
	def __init__(self,
				synthesizer_weights='logs/'):
		super(Generator, self).__init__()

		self.synthesizer = sif.Synthesizer(synthesizer_weights, verbose=False, manual_inference=True)

	def vc(self, sample, id):
		id_windows = [range(i, i + sif.hparams.T) for i in range(0, sample['till'], 
					sif.hparams.T - sif.hparams.overlap) if (i + sif.hparams.T <= sample['till'])]

		all_images = [[sample['folder'].format(id) for id in window] for window in id_windows]
		ref = np.load(os.path.join(os.path.dirname(sample['folder']), 'ref.npz'))['ref'][0]
		ref = np.expand_dims(ref, 0)
		cnt = 0
		for window_fnames in tqdm(all_images):
			window = []
			for fname in window_fnames:
				img = cv2.imread(fname)
				img = cv2.resize(img, (sif.hparams.img_size, sif.hparams.img_size))
				window.append(img)

			images = np.asarray(window) / 255. # T x H x W x 3

			s = self.synthesizer.synthesize_spectrograms(images, ref)[0]
			if cnt == 0:
				mel = s
			else:
				mel = np.concatenate((mel, s[:, sif.hparams.mel_overlap:]), axis=1)
			cnt = cnt + 1

		wav = self.synthesizer.griffin_lim(mel)
		sif.audio.save_wav(wav, 'logs/wavs/out{}.wav'.format(id), sr=sif.hparams.sample_rate)

samples = [{'folder' : '../test_data/LIVING_00392/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
			'gt' : '../test_data/LIVING_00392/mels.npz'},

			{'folder' : '../test_data/LIVING_00613/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/LIVING_00613/mels.npz'},

			{'folder' : '../test_data/MAJOR_00551/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/MAJOR_00551/mels.npz'},

			{'folder' : '../test_data/MAJOR_00698/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/MAJOR_00698/mels.npz'},

			{'folder' : '../test_data/MEASURES_00027/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/MEASURES_00027/mels.npz'},

			{'folder' : '../test_data/MEASURES_00077/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/MEASURES_00077/mels.npz'},

			{'folder' : '../test_data/MEETING_00005/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/MEETING_00005/mels.npz'},

			{'folder' : '../test_data/SENSE_00076/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/SENSE_00076/mels.npz'},

			{'folder' : '../test_data/SENSE_00561/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/SENSE_00561/mels.npz'},

			{'folder' : '../test_data/WELCOME_00788/{}.jpg', 'till' : (29//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/WELCOME_00788/mels.npz'}]

ids = [i + 1 for i in range(len(samples))]

if __name__ == '__main__':
	g = Generator()
	for s, id in zip(samples, ids):
		g.vc(s, id)