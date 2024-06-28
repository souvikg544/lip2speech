from scipy.io import wavfile
from pypesq import pypesq
from pystoi.stoi import stoi
from glob import glob
import os, librosa, sys, pickle
import numpy as np
from tqdm import tqdm

dataset = sys.argv[1]
model = sys.argv[2]
sr = 16000

all_files = glob("{}/{}/*.wav".format(dataset, model))
gt_folder = dataset + '/gts/{}'

print('Calculating for {} files'.format(len(all_files)))

total_stoi = 0
total_pesq = 0
total_estoi = 0

prog_bar = tqdm(all_files)

pairs = []

for idx, filename in enumerate(prog_bar):
	file_id = os.path.basename(filename)
	gt_filename = gt_folder.format(file_id)

	# print('Calculating for {} with gt file: {}'.format(filename, gt_filename) )
	rate, deg = wavfile.read(filename)
	rate, ref = wavfile.read(gt_filename)

	if rate != sr:
		ref = librosa.resample(ref.astype(np.float32), rate, sr).astype(np.int16)
		rate = sr

	if len(ref) > len(deg): x = ref[0 : deg.shape[0]]
	elif len(deg) > len(ref):
		deg = deg[: ref.shape[0]]
		x = ref

	total_pesq += pypesq(rate, x, deg, 'wb')
	total_stoi += stoi(x, deg, rate, extended=False)
	estoi = stoi(x, deg, rate, extended=True)
	total_estoi += estoi

	prog_bar.set_description('{}'.format(total_estoi / (idx + 1)))
	pairs.append((estoi, filename))

sortedpairs = sorted(pairs, reverse=True)
print(sortedpairs[:20])
pickle.dump(sortedpairs, open('sortedpairs', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

print('Mean PESQ: {}'.format(total_pesq / len(all_files)))
print('Mean STOI: {}'.format(total_stoi / len(all_files)))
print('Mean ESTOI: {}'.format(total_estoi / len(all_files)))
