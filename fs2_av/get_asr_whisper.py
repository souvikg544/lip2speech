import argparse
import whisper
import numpy as np
import subprocess, os
from glob import glob 
from tqdm import tqdm

model = whisper.load_model("medium")

def transcribe(args, audio_file, result_path):

	result = model.transcribe(audio_file)
	result = result["text"]
	# print(result)

	pred_fname = os.path.join(result_path, "pred_text", os.path.basename(audio_file).split(".")[0]+".txt")
	# print("Pred fname: ", pred_fname)

	gt_fname = audio_file.split("/")[-1].split(".")[0].split("_")[0] + "/" + audio_file.split("/")[-1].split(".")[0].split("_")[1]
	gt_src_fname = os.path.join(args.gt_text_path, gt_fname+".txt")
	# print("GT src fname: ", gt_src_fname)
	gt_dest_fname = os.path.join(result_path, "gt_text", os.path.basename(audio_file).split(".")[0]+".txt")
	# print("GT dest fname: ", gt_dest_fname)
	
	with open(pred_fname, "w") as f:
		f.write(result)

	subprocess.call("rsync -az {} {}".format(gt_src_fname, gt_dest_fname), shell=True)

	return result

def load_data(args):

	audio_paths = sorted(list(glob('{}/*.wav'.format(args.audio_path))))
	print("Total audio files = ", len(audio_paths))
	
	result_path = args.result_path
	os.makedirs(os.path.join(result_path, "pred_text"), exist_ok=True)
	os.makedirs(os.path.join(result_path, "gt_text"), exist_ok=True)

	
	prog_bar = tqdm(audio_paths)

	
	for audio in prog_bar:

		try:
			pred_text = transcribe(args, audio, result_path)
		except:
			continue


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--audio_path", required=True)
	parser.add_argument("--result_path", default="results", type=str, help="folder path to save the results")
	parser.add_argument("--gt_text_path")    
	args = parser.parse_args()

	load_data(args)