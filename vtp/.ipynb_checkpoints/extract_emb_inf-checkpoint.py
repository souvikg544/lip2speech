import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import load_args
from models import builders

from dataloader import AugmentationPipeline, VideoDataset
from utils import load

from glob import glob

from torch.cuda.amp import autocast


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def init(args):
	data_util = VideoDataset(args, mode='val', inference=True)

	model = builders[args.builder](data_util.vocab_size + 1, args.feat_dim, N=args.num_blocks, 
							d_model=args.hidden_units, 
							h=args.num_heads, dropout=args.dropout_rate)

	return model.to(args.device).eval(), data_util

def save_feat(vidpath, featpath, model, data_util):
	

	src = torch.FloatTensor(data_util.input_transform(data_util.read_video(vidpath), 
							augment=False)).unsqueeze(0)

	src = src.permute(0, 4, 1, 2, 3)
	src_mask = torch.ones((1, 1, src.size(2)))
	with torch.no_grad(): 
		src = augmentor(src, train_mode=False).detach()

	with torch.no_grad():
		with autocast():
			src = src.cuda()
			src_mask = src_mask.cuda()
			outs = []
			chunk_size = 512 # increase or decrease based on how much GPU memory you have
			i = 0
			while i < src.size(2):
				s = src[:, :, i : i + chunk_size]
				m = src_mask[:, :, i : i + chunk_size]

				outs.append(model.face_encoder(s, m)[0].cpu())

				i += chunk_size

			out = torch.cat(outs, dim=0)

			np.save(featpath, out.cpu().numpy().astype(np.float16))
			print("Extracted VTP embeddings: ", out.cpu().numpy().shape)
			# exit(0)


def main(args):
	args.device = 'cuda'
	model, data_util = init(args)
	if args.ckpt_path is not None:
		# print('Resuming from: {}'.format(args.ckpt_path))
		model = load(model, args.ckpt_path)[0]
	else:
		raise SystemError('Need a checkpoint to dump feats')

	
	save_feat(args.videos_root, args.feats_root, model, data_util)


def save_visual_emb(vidpath, featpath, ckpt_path=None):
	
	args = load_args()
	augmentor = AugmentationPipeline(args)

	args.device = 'cuda'
	model, data_util = init(args)
	if ckpt_path is not None:
		model = load(model, ckpt_path)[0]
	else:
		raise SystemError('Need a checkpoint to dump feats')

	src = torch.FloatTensor(data_util.input_transform(data_util.read_video(vidpath), 
							augment=False)).unsqueeze(0)

	src = src.permute(0, 4, 1, 2, 3)
	src_mask = torch.ones((1, 1, src.size(2)))
	with torch.no_grad(): 
		src = augmentor(src, train_mode=False).detach()

	with torch.no_grad():
		with autocast():
			src = src.cuda()
			src_mask = src_mask.cuda()
			outs = []
			chunk_size = 512 # increase or decrease based on how much GPU memory you have
			i = 0
			while i < src.size(2):
				s = src[:, :, i : i + chunk_size]
				m = src_mask[:, :, i : i + chunk_size]

				outs.append(model.face_encoder(s, m)[0].cpu())

				i += chunk_size

			out = torch.cat(outs, dim=0)

			np.save(featpath, out.cpu().numpy().astype(np.float16))
			print("Extracted VTP embeddings: ", out.cpu().numpy().shape)

if __name__ == '__main__':
	args = load_args()
	augmentor = AugmentationPipeline(args)
	main(args)