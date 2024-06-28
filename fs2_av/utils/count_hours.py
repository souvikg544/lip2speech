import argparse
import os
from tqdm import tqdm
import random
from glob import glob
from decord import VideoReader
from decord import cpu, gpu

def count(videos):

	counts = []

	progress_bar = tqdm(videos)
	for video in progress_bar:

		try:
			vr = VideoReader(video, ctx=cpu(0))
		except:
			continue

		counts.append(len(vr))		

		progress_bar.set_description("# Frames: %.4f, # Hours: %.4f" % (sum(counts), ((sum(counts)/25)/3600)))

	print("Total num of frames = ", sum(counts))
	print("Average num of frames = ", sum(counts)/len(counts))
	print("Total num of hours = ", (sum(counts)/25)/3600)

def create_subset(videos):

	counts = []
	file = open("vox2_subset3.txt", "w")

	random.shuffle(videos)
	progress_bar = tqdm(videos)

	for video in progress_bar:

		try:
			vr = VideoReader(video, ctx=cpu(0))
		except:
			continue

		counts.append(len(vr))		

		hours = (sum(counts)/25)/3600

		if hours < 200:
			name = video.split("/")[-3] + "/" + video.split("/")[-2] + "/" + video.split("/")[-1]
			# print(name)
			file.write(name)
			file.write("\n")
		else:
			break

		progress_bar.set_description("# Frames: %.4f, # Hours: %.4f" % (sum(counts), ((sum(counts)/25)/3600)))

	print("Total num of frames = ", sum(counts))
	print("Average num of frames = ", sum(counts)/len(counts))
	print("Total num of hours = ", (sum(counts)/25)/3600)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-p', '--data_path', required=True, help='Path containing the videos')	
	
	args = parser.parse_args()

	videos = glob(os.path.join("{}/*/*.mp4".format(args.data_path)))
	print("No of videos: ", len(videos))

	count(videos)
	# create_subset(videos)