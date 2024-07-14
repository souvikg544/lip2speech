# Official Implementation of Towards Accurate Lip-to-Speech Synthesis in-the-Wild
### Abstract
In this paper, we introduce a novel approach to address the task of synthesizing speech from silent videos of any in-the-wild speaker solely based on lip movements. The traditional approach of directly generating speech from lip videos faces the challenge of not being able to learn a robust language model from speech alone, resulting in unsatisfactory outcomes. To overcome this issue, we propose incorporating noisy text supervision using a state-of-the-art lip-to-text network that instills language information into our model. The noisy text is generated using a pre-trained lip-to-text model, enabling our approach to work without text annotations during inference. We design a visual text-to-speech network that utilizes the visual stream to generate accurate speech, which is in-sync with the silent input video. We perform extensive experiments and ablation studies, demonstrating our approach's superiority over the current state-of-the-art methods on various benchmark datasets. Further, we demonstrate an essential practical application of our method in assistive technology by generating speech for an ALS patient who has lost the voice but can make mouth movements. Our demo video, code, and additional details can be found at http://cvit.iiit.ac.in/research/projects/cvit-projects/ms-l2s-itw.

## Data Preparation

1. Speaker Embeddings
```
cd speaker_emb
python preprocess_speakers_mp4.py --final_data_root path/to/your/dataset
```

2. Phonemes Generation
```
cd fs2_av/utils/
python save_phonemes.py --data_path /ssd_scratch/cvit/souvik/mvlrs_v1/pretrain/ --preprocess_config /home2/souvikg544/souvik/lip2speech/fs2_av/config/LRS_train/preprocess.yaml

```

3. VTP Embeddings
```
cd vtp
python run_pipeline.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python extract_feats.py --builder vtp24x24 --ckpt_path <path_to_a_feat_extractor> --videos_root <video_data_root> --file_list */*.mp4 --feats_root <feature_extraction_dest_root>

```

The resulting dataset should look like :
data_root/<speakerid>/
| - 00001.mp4
| - 00001.wav
| - 00001.txt
| - 00001_vtp.np
| - 00001.npz


## Training
```
cd fs2_av
python train_all.py --preprocess_config config/LRS_train/preprocess.yaml --model_config config/LRS_train/model.yaml --train_config config/LRS_train/train.yaml

```

## License and Citation
```
@inproceedings{10.1145/3581783.3611787,
author = {Hegde, Sindhu and Mukhopadhyay, Rudrabha and Jawahar, C.V and Namboodiri, Vinay},
title = {Towards Accurate Lip-to-Speech Synthesis in-the-Wild},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3611787},
doi = {10.1145/3581783.3611787},
numpages = {9},
keywords = {assistive technology, lip-reading, lip-to-speech, speech generation},
location = {Ottawa ON, Canada},
series = {MM '23}
}
```