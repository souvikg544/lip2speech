import numpy as np
import argparse

def find_common(args, split="test"):
	fname = 'utils/filelists/{}.txt'.format(split)
	files_lrs2 = np.loadtxt(fname, str, delimiter=" ")

	filelist_lrs2_test = []
	for i in range(len(files_lrs2)):
	# for i in range(2):
		file = files_lrs2[i][0]
		filelist_lrs2_test.append(file.split("/")[0] + "_" + file.split("/")[1])

	files_fs2 = args.fs2_file
	filelist_fs2_val = []
	# i=0
	with open(files_fs2, "r", encoding="utf-8") as f:
		for line in f.readlines():
			folder, speaker, phoneme, raw_text = line.strip("\n").split("|")
			filelist_fs2_val.append(str(speaker) + "_" + folder)
			# i+=1
			# if i>2:
			# 	break

	# print("LRS2 official: ", filelist_lrs2_test)
	# print("FS2 val: ", filelist_fs2_val)

	common_files = list(set(filelist_lrs2_test) & set(filelist_fs2_val))
	print("No of common files = ", len(common_files))
	print("Common files: ", common_files)

	return common_files

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--fs2_file", type=str, help="file path of the fs2 val set")
	args = parser.parse_args()
	
	common_files = find_common(args)