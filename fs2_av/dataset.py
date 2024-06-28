import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D

from decord import VideoReader
from decord import cpu, gpu

import random

import librosa
import audio as Audio
import audio.hparams as hp
import audio.audio_utils as au

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.raw_data_path = preprocess_config["path"]["corpus_path"]
        self.img_size = preprocess_config["preprocessing"]["img_size"]
        self.scale_factor = preprocess_config["preprocessing"]["upsample_scale"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        
        while(1):
            idx = random.randint(0, len(self.text) - 1)
            basename = self.basename[idx]
            # print("Basename: ", basename)

            speaker = self.speaker[idx]
            # print("Speaker: ", speaker)
            
            speaker_id = self.speaker_map[speaker]

            raw_text = self.raw_text[idx]

            phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

            mel_path = os.path.join(
                self.preprocessed_path,
                "mel",
                "{}-mel-{}.npy".format(speaker, basename),
            )
            mel = np.load(mel_path)
            # print("Mels: ", mel.shape)

            pitch_path = os.path.join(
                self.preprocessed_path,
                "pitch",
                "{}-pitch-{}.npy".format(speaker, basename),
            )
            pitch = np.load(pitch_path)

            energy_path = os.path.join(
                self.preprocessed_path,
                "energy",
                "{}-energy-{}.npy".format(speaker, basename),
            )
            energy = np.load(energy_path)

            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)

            
            video_file = os.path.join(self.raw_data_path, speaker, basename)+'.mp4'
            visual_emb, frames_mask, frames = self.load_frames(video_file, test=False)

            if visual_emb is None:
                continue
                # visual_emb = np.zeros_like((10,512))

            # wav_path = os.path.join(self.raw_data_path, speaker, basename)+'.wav'
            # wav, _ = librosa.load(wav_path, sr=16000)
            # wav = wav[:16000]
            # mel_spectrogram = au.melspectrogram(wav, hp.hparams).T[:-1]
            # mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, self.STFT)
            # print("N Frames: ", len(frames), "\tVisual emb shape: ", visual_emb.shape, "\tMel shape: ", mel.shape, "\tMel on-the-fly shape: ", mel_spectrogram.shape)

            try:
                se_path = os.path.join(self.raw_data_path, speaker, basename)+'.npz'
                speaker_emb = np.load(se_path)['ref'][0]
            except:
                continue
                # speaker_emb = np.zeros_like((256,))
            # print("Speaker emb: ", speaker_emb.shape)
            
            sample = {
                "id": basename,
                "speaker": speaker_id,
                "text": phone,
                "raw_text": raw_text,
                "mel": mel,
                "pitch": pitch,
                "energy": energy,
                "duration": duration,
                "visual_emb": visual_emb,
                "speaker_emb": speaker_emb,
            }
            # print("Sample: ", sample)
            

            return sample

    def load_frames(self, video_file, test):

        try:
            visual_emb_file = video_file.replace(".mp4", "_vtp.npy")
            visual_emb = np.load(visual_emb_file)
        except:
            return None, None, None
        # print("Visual emb: ", visual_emb.shape)                     # n_framesx512

        frames_mask = np.ones(visual_emb.shape[0])

        frames=None
        if test:
            try:
                vr = VideoReader(video_file, ctx=cpu(0), width=self.img_size, height=self.img_size)
            except:
                return None, None, None

            frames = [vr[i].asnumpy() for i in range(len(vr))]
            frames = np.asarray(frames)  / 255.

        return visual_emb, frames_mask, frames

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        visual_emb = [data[idx]["visual_emb"] for idx in idxs]
        # frames = [data[idx]["frames"] for idx in idxs]
        # frames_mask = [data[idx]["frames_mask"] for idx in idxs]
        speaker_emb = [data[idx]["speaker_emb"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        video_lens = np.array([emb.shape[0] for emb in visual_emb])
        mel_lens_video = np.array([int(vid_len*self.scale_factor) for vid_len in video_lens])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        visual_emb = pad_2D(visual_emb)
        speaker_emb = np.array(speaker_emb)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            visual_emb,
            video_lens,
            max(video_lens),
            mel_lens_video,
            max(mel_lens_video),
            speaker_emb,
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,

        )

    def collate_fn(self, data):
        data_size = len(data)
        # print("data size: ", data_size)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        # print("IDX ARR:", len(idx_arr))
        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        # print("Output: ", output)
        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )