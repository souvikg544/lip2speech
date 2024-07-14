import argparse
import os
from tqdm import tqdm
import random
from glob import glob
from g2p_en import G2p
from string import punctuation
import re
import yaml
import numpy as np
import sys
sys.path.append('../')
from text import text_to_sequence

g2p = G2p()


def save_phonemes(text_files, preprocess_config):

	progress_bar = tqdm(text_files)
	for file in progress_bar:
		# print("File: ", file)

		phoneme_fname = file.replace(".txt", "_phonemes.npy")
		if os.path.exists(phoneme_fname):
			continue
		try:
			with open(file) as f:
				raw_texts = f.read().splitlines()
			raw_texts = raw_texts[0].split(":  ")[1]
		except:
			continue   
		phonemes = preprocess_english(raw_texts, preprocess_config)
		print(phonemes,phoneme_fname)
		np.save(phoneme_fname, phonemes)
        
def read_lexicon(lex_path):
	lexicon = {}
	with open(lex_path) as f:
		for line in f:
			temp = re.split(r"\s+", line.strip("\n"))
			word = temp[0]
			phones = temp[1:]
			if word.lower() not in lexicon:
				lexicon[word.lower()] = phones
	return lexicon


def preprocess_english(text, preprocess_config):
	text = text.rstrip(punctuation)
	lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
	
	phones = []
	words = re.split(r"([,;.\-\?\!\s+])", text)
	for w in words:
		if w.lower() in lexicon:
			phones += lexicon[w.lower()]
		else:
			phones += list(filter(lambda p: p != " ", g2p(w)))
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

if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-p', '--data_path', required=True, help='Path containing the videos')
	parser.add_argument("--preprocess_config", type=str, required=True, help="path to preprocess.yaml")

	args = parser.parse_args()

	text_files = glob(os.path.join("{}/*/*.txt".format(args.data_path)))
	print("No of text files: ", len(text_files))

	preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
	# count(videos)
	# create_subset(videos)
	save_phonemes(text_files, preprocess_config)