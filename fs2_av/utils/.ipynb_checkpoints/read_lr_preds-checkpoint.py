import numpy as np
import pickle
import argparse

def read_file(file):
	# with open(file, 'rb') as p:
	# 	pred_dict = pickle.load(p)

	# with open(args.text_file) as f:
		# raw_data = f.read().splitlines()

	raw_data = np.loadtxt(file, str, delimiter=',')

	# print("Data: ", raw_data[0])
	text_dict = {}
	for i in range(len(raw_data)):
		key = raw_data[i][0]
		value = raw_data[i][1]
		text_dict[key] = value

	print(text_dict)
	# print("Len: ", len(pred_dict))
	# print("Keys: ", pred_dict.keys())
	# text = pred_dict["lrs2/vid/5984065405828502341/00020.npy"]
	# print(text)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--text_file", type=str, help="file path of the predictions")
	args = parser.parse_args()
	
	read_file(args.text_file)