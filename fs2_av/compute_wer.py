import argparse, os, random
import numpy as np
from glob import glob
from tqdm import tqdm
# from utils.text_utils import *

def levenshtein(a, b):
	"""Calculates the Levenshtein distance between a and b.
	The code was taken from: http://hetland.org/coding/python/levenshtein.py
	"""
	n, m = len(a), len(b)
	if n > m:
		# Make sure n <= m, to use O(min(n,m)) space
		a, b = b, a
		n, m = m, n
	current = list(range(n + 1))
	for i in range(1, m + 1):
		previous, current = current, [i] + [0] * n
		for j in range(1, n + 1):
			add, delete = previous[j] + 1, current[j - 1] + 1
			change = previous[j - 1]
			if a[j - 1] != b[i - 1]:
				change = change + 1
			current[j] = min(add, delete, change)
	return current[n]

def calculate_wer(args):

	text_paths = sorted(list(glob('{}/*.txt'.format(args.text_path))))
	print("Total text files = ", len(text_paths))

	total_wer, total_cer, total_tokens, total_chars = 0., 0., 0., 0.
	# total_wer, total_cer = [], []
	prog_bar = tqdm(text_paths)

	for pred_file in prog_bar:

		gt_file = pred_file.replace("pred_text", "gt_text")

		with open(pred_file) as f:
			pred = f.read().splitlines()
			pred = pred[0][1:].lower()
			pred = pred.replace(",", "")
			pred = pred.replace(".", "")
			# pred = pred.replace("'", "")
			
		with open(gt_file) as f:
			gt = f.read().splitlines()
			gt = gt[0].split(":  ")[1].lower()
			# gt = gt.replace("'", "")
			
		
		print("GT : ", gt, "| Pred: ", pred)
		wer = levenshtein(gt.split(), pred.split())
		cer = levenshtein(list(gt), list(pred))
		# wer = wer_(gt, pred)
		# cer = cer_(gt, pred)

		total_wer += wer
		total_cer += cer
		total_tokens += len(gt.split())
		total_chars += len(list(gt))
		# total_wer.append(wer)
		# total_cer.append(cer)

		prog_bar.set_description('WER: {}, CER: {}'.format(
								wer, cer))

	print("WER = ", total_wer / total_tokens)
	print("CER = ", total_cer / total_chars)
	# print("WER = ", sum(total_wer) / len(total_wer))
	# print("CER = ", sum(total_cer) / len(total_cer))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--text_path", required=True)
	args = parser.parse_args()

	calculate_wer(args)