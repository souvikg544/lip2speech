import argparse
import numpy as np
from glob import glob
import os, sys
import subprocess
from tqdm import tqdm

def remove_folders(folders, pretrain_folders):

	count = 0
	for folder in tqdm(folders):

		if folder in pretrain_folders:
			print(count)
			count+=1
			# subprocess.call('rm -r {}'.format(folder), shell=True)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-p', '--data_path', required=True, help='Path of the pretrain set')
	parser.add_argument('-f', '--file', required=True, help='Path containing the filenames to remove from the main set')

	args = parser.parse_args()

	pretrain_folders = glob('{}/*'.format(args.data_path))
	print("No of pretrain folders: ", len(pretrain_folders))

	folders = np.loadtxt(args.file, str)
	print("No of folders to remove: ", len(folders))

	remove_folders(folders, pretrain_folders)