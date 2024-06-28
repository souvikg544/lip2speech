import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

import cv2, subprocess
import numpy as np

matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device, train=True):

    if train:
        if len(data) == 16:
            (ids, raw_texts, speakers, texts, src_lens, max_src_len, visual_emb, video_lens, max_video_len, mel_lens_video,\
                max_mel_len_video, speaker_emb, mels, mel_lens, max_mel_len, frames) = data

            speakers = torch.from_numpy(speakers).long().to(device)
            texts = torch.from_numpy(texts).long().to(device)
            src_lens = torch.from_numpy(src_lens).to(device)
            mels = torch.from_numpy(mels).float().to(device)
            visual_emb = torch.from_numpy(visual_emb).float().to(device)
            speaker_emb = torch.from_numpy(speaker_emb).float().to(device)
            video_lens = torch.from_numpy(video_lens).to(device)
            mel_lens_video = torch.from_numpy(mel_lens_video).to(device)
            mel_lens = torch.from_numpy(mel_lens).to(device)
            # frames = torch.from_numpy(frames).to(device)

            return (ids, raw_texts, speakers, texts, src_lens, max_src_len, visual_emb, video_lens, max_video_len, \
                mel_lens_video, max_mel_len_video, speaker_emb, mels, mel_lens, max_mel_len, frames)

        if len(data) == 13:
            (texts, src_lens, max_src_len, visual_emb, video_lens, max_video_len, mel_lens_video, max_mel_len_video, \
                speaker_emb, mels, mel_lens, max_mel_len, frames) = data

            texts = torch.from_numpy(texts).long().to(device)
            src_lens = torch.from_numpy(src_lens).to(device)
            mels = torch.from_numpy(mels).float().to(device)
            visual_emb = torch.from_numpy(visual_emb).float().to(device)
            speaker_emb = torch.from_numpy(speaker_emb).float().to(device)
            video_lens = torch.from_numpy(video_lens).to(device)
            mel_lens_video = torch.from_numpy(mel_lens_video).to(device)
            mel_lens = torch.from_numpy(mel_lens).to(device)

            return (texts, src_lens, max_src_len, visual_emb, video_lens, max_video_len, mel_lens_video, max_mel_len_video, \
                    speaker_emb, mels, mel_lens, max_mel_len, frames)

    else:
        if len(data) == 13:
            (ids, raw_texts, speakers, texts, src_lens, max_src_len, visual_emb, \
                video_lens, max_video_len, mel_lens_video, max_mel_len_video, speaker_emb, frames) = data

            speakers = torch.from_numpy(speakers).long().to(device)
            texts = torch.from_numpy(texts).long().to(device)
            src_lens = torch.from_numpy(src_lens).to(device)
            visual_emb = torch.from_numpy(visual_emb).float().to(device)
            speaker_emb = torch.from_numpy(speaker_emb).float().to(device)
            video_lens = torch.from_numpy(video_lens).to(device)
            mel_lens_video = torch.from_numpy(mel_lens_video).to(device)

            return (ids, raw_texts, speakers, texts, src_lens, max_src_len, visual_emb, \
                    video_lens, max_video_len, mel_lens_video, max_mel_len_video, speaker_emb, frames)

        elif len(data) == 10:
            (texts, src_lens, max_src_len, visual_emb, video_lens, max_video_len, \
                mel_lens_video, max_mel_len_video, speaker_emb, frames) = data

            texts = torch.from_numpy(texts).long().to(device)
            src_lens = torch.from_numpy(src_lens).to(device)
            visual_emb = torch.from_numpy(visual_emb).float().to(device)
            speaker_emb = torch.from_numpy(speaker_emb).float().to(device)
            video_lens = torch.from_numpy(video_lens).to(device)
            mel_lens_video = torch.from_numpy(mel_lens_video).to(device)

            return (texts, src_lens, max_src_len, visual_emb, video_lens, max_video_len, \
                mel_lens_video, max_mel_len_video, speaker_emb, frames)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        # logger.add_scalar("Loss/pitch_loss", losses[3], step)
        # logger.add_scalar("Loss/energy_loss", losses[4], step)
        # logger.add_scalar("Loss/duration_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)

def save_video(gt_wav_fpath, wav_fpath, input_frames, video_fpath):
                
    # Save GT video
    gt_vid_fname = str(video_fpath)+'_gt'
    generate_video(input_frames, gt_wav_fpath, gt_vid_fname)

    # Save generated video
    generated_vid_fname = str(video_fpath)+'_pred'
    generate_video(input_frames, wav_fpath, generated_vid_fname)     
    
def generate_video(frames, audio_file, output_file_name, fps=25):

    fname = 'output.avi'
    video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frames[0].shape[1], frames[0].shape[0]))
 
    # frame_fpath = output_file_name
    # if not os.path.exists(frame_fpath): 
    #     os.mkdir(frame_fpath)
    for i in range(len(frames)):
        # print(frames[i].shape)
        # img = np.clip(np.round(frames[i]*255), 0, 255)
        img = np.clip(np.round(frames[i]), 0, 255)
        # print(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(img.shape)
        # fname=os.path.join("results/trial", str(i)+'.png')
        # print(fname)
        # cv2.imwrite(fname, np.uint8(img))
        video.write(np.uint8(img))
    
    video.release()

    no_sound_video = str(output_file_name) + '_nosound.mp4'
    subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -c copy -an -strict -2 %s' % (fname, no_sound_video), shell=True)

    video_output = str(output_file_name) + '.mp4'
    subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 %s' % 
                    (audio_file, no_sound_video, video_output), shell=True)

    os.remove(fname)
    os.remove(no_sound_video)

def generate_video_inference(frames, audio_file, output_file_name, fps=25):

    fname = 'inference.avi'
    video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frames[0].shape[1], frames[0].shape[0]))
 
    # frame_fpath = output_file_name
    # if not os.path.exists(frame_fpath): 
    #     os.mkdir(frame_fpath)
    for i in range(len(frames)):
        # print(frames[i].shape)
        # img = np.clip(np.round(frames[i]*255), 0, 255)
        # print(img)
        img = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        # print(img.shape)
        # fname=os.path.join("results/trial", str(i)+'.png')
        # print(fname)
        # cv2.imwrite(fname, np.uint8(img))
        video.write(np.uint8(img))
    
    video.release()

    no_sound_video = str(output_file_name) + '_nosound.mp4'
    subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -c copy -an -strict -2 %s' % (fname, no_sound_video), shell=True)

    video_output = str(output_file_name) + '.mp4'
    subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 %s' % 
                    (audio_file, no_sound_video, video_output), shell=True)

    os.remove(fname)
    os.remove(no_sound_video)

def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    # basename = targets[0][0]
    # src_len = predictions[8][0].item()
    mel_len = predictions[5][0].item()
    mel_target = targets[9][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    # duration = targets[11][0, :src_len].detach().cpu().numpy()
    # if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
    #     pitch = targets[9][0, :src_len].detach().cpu().numpy()
    #     pitch = expand(pitch, duration)
    # else:
    #     pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    # if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
    #     energy = targets[10][0, :src_len].detach().cpu().numpy()
    #     energy = expand(energy, duration)
    # else:
    #     energy = targets[10][0, :mel_len].detach().cpu().numpy()

    # with open(
    #     os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    # ) as f:
    #     stats = json.load(f)
    #     stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), None, None),
            (mel_target.cpu().numpy(), None, None),
        ],
        None,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer, vocoder_infer_mel

        wav_reconstruction = vocoder_infer_mel(
            mel_target.unsqueeze(0),
            model_config,
            preprocess_config,
            
        )
        wav_prediction = vocoder_infer_mel(
            mel_prediction.unsqueeze(0),
            model_config,
            preprocess_config,
            
        )
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction

def synth_one_sample_all(targets, predictions, vocoder, model_config, preprocess_config):

    mel_len = predictions[5][0].item()
    mel_target = targets[9][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    
    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), None, None),
            (mel_target.cpu().numpy(), None, None),
        ],
        None,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer, vocoder_infer_mel, vocoder_infer_mel_all

        wav_reconstruction = vocoder_infer_mel_all(
            mel_target,
            model_config,
            preprocess_config,
            
        )
        wav_prediction = vocoder_infer_mel_all(
            mel_prediction,
            model_config,
            preprocess_config,
            
        )
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction

def synth_one_sample_all_vocoder(targets, predictions, vocoder, model_config, preprocess_config):

    mel_len = predictions[5][0].item()
    mel_target = targets[9][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    
    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), None, None),
            (mel_target.cpu().numpy(), None, None),
        ],
        None,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer, vocoder_infer_mel, vocoder_infer_mel_all, vocoder_infer_hifigan

        wav_reconstruction = vocoder_infer_hifigan(
            mel_target,
            vocoder,
            model_config,
            preprocess_config,
            
        )[0]
        wav_prediction = vocoder_infer_hifigan(
            mel_prediction,
            vocoder,
            model_config,
            preprocess_config,
            
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction

def synth_one_sample_inference(targets, predictions, vocoder, model_config, preprocess_config, path, result_fname):

    basename = targets[0]
    mel_len = predictions[5][0].item()
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    pred_mel_path = os.path.join(path, "{}.npy".format(result_fname))
    np.save(pred_mel_path, mel_prediction.cpu().numpy())
    print("Saved the melspectrogram prediction at: ", pred_mel_path)

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), None, None),
        ],
        None,
        ["Synthetized Spectrogram"],
    )
    # plt.savefig(os.path.join(path, "{}.png".format(basename)))
    plt.savefig(os.path.join(path, "melspec.png"))
    plt.close()

    from .model import vocoder_infer, vocoder_infer_mel, vocoder_infer_mel_all, vocoder_infer_hifigan

    wav_prediction = vocoder_infer_mel(
        mel_prediction.unsqueeze(0),
        model_config,
        preprocess_config,        
    )
    # wav_prediction = vocoder_infer_hifigan(
    #     mel_prediction.unsqueeze(0),
    #     vocoder,
    #     model_config,
    #     preprocess_config,        
    # )[0]

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    pred_wav_fpath = os.path.join(path, "{}.wav".format(result_fname))
    # pred_wav_fpath = os.path.join(path, "result.wav")
    wavfile.write(pred_wav_fpath, sampling_rate, wav_prediction)
    print("Saved the speech file: ", pred_wav_fpath)

    video_fpath = os.path.join(path, "{}".format(result_fname))
    # video_fpath = os.path.join(path, "result")
    frames = targets[-1]
    print("Frames: ", frames.shape)
    generate_video_inference(frames, pred_wav_fpath, video_fpath)
    print("Saved the video at: ", str(video_fpath)+".mp4")


def synth_sample_generation(targets, predictions, model_config, preprocess_config):

    mel_len = predictions[5][0].item()
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)

    from .model import vocoder_infer_mel

    wav_prediction = vocoder_infer_mel(
        mel_prediction.unsqueeze(0),
        model_config,
        preprocess_config,        
    )

    return wav_prediction, mel_prediction


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        # wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)
        wavfile.write(os.path.join("output", "result", "{}.wav".format(basename)), sampling_rate, wav)
        print("Saved the speech file: ", os.path.join("output", "result", "{}.wav".format(basename)))


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    # pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    # pitch_min = pitch_min * pitch_std + pitch_mean
    # pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        # pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        # ax1 = add_axis(fig, axes[i][0])
        # ax1.plot(pitch, color="tomato")
        # ax1.set_xlim(0, mel.shape[1])
        # ax1.set_ylim(0, pitch_max)
        # ax1.set_ylabel("F0", color="tomato")
        # ax1.tick_params(
        #     labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        # )

        # ax2 = add_axis(fig, axes[i][0])
        # ax2.plot(energy, color="darkviolet")
        # ax2.set_xlim(0, mel.shape[1])
        # ax2.set_ylim(energy_min, energy_max)
        # ax2.set_ylabel("Energy", color="darkviolet")
        # ax2.yaxis.set_label_position("right")
        # ax2.tick_params(
        #     labelsize="x-small",
        #     colors="darkviolet",
        #     bottom=False,
        #     labelbottom=False,
        #     left=False,
        #     labelleft=False,
        #     right=True,
        #     labelright=True,
        # )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
