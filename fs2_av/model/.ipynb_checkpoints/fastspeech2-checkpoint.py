import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import *
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.text_encoder = TextEncoder(model_config)
        self.visual_encoder = VisualEncoder(model_config)
        self.text_video_aligner = ScaledDotProductAttention(model_config)
        self.scale_factor = model_config["transformer"]["upsample_scale"]
        self.upsampler = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        # self.speaker_emb = None
        # if model_config["multi_speaker"]:
        #     with open(
        #         os.path.join(
        #             preprocess_config["path"]["preprocessed_path"], "speakers.json"
        #         ),
        #         "r",
        #     ) as f:
        #         n_speaker = len(json.load(f))
        #     self.speaker_emb = nn.Embedding(
        #         n_speaker,
        #         model_config["transformer"]["encoder_hidden"],
        #     )

    def forward(
        self,
        speakers,
        texts,
        text_lens,
        max_text_len,
        visual_emb,
        video_lens,
        max_video_len,
        mel_lens_video,
        max_mel_len_video,
        speaker_emb,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        # print("Visual emb: ", visual_emb.shape)
        # print("Speaker emb: ", speaker_emb.shape)

        text_masks = get_mask_from_lengths(text_lens, max_text_len)
        # print("Text mask: ", text_masks.shape)

        text_output = self.text_encoder(texts, text_masks)
        # print("Text encoder output: ", text_output.shape)

        visual_masks = get_mask_from_lengths(video_lens, max_video_len)
        # print("Video mask: ", visual_masks.shape)

        visual_output = self.visual_encoder(visual_emb, visual_masks)
        # print("Visual encoder output: ", visual_output.shape)

        output, attn_wts = self.text_video_aligner(visual_output, text_output, text_output)
        # print("Attn Output: ", output.shape)
        # print("Attn weights: ", attn_wts.shape)

        h_context = output + visual_output
        # print("H context: ", h_context.shape)
        h_mel = self.upsampler(h_context.permute(0,2,1)).permute(0,2,1)
        # print("H mels: ", h_mel.shape)

        repeat_len = h_mel.size(1)
        output = h_mel + speaker_emb.unsqueeze(1).expand(-1, repeat_len, -1)
        # print("Input to variance adaptor: ", output.shape)
        # print("Max mel len: ", max_mel_len)

        # mel_masks = (
        #     get_mask_from_lengths(mel_lens, max_mel_len)
        #     if mel_lens is not None
        #     else None
        # )

        mel_mask_video = get_mask_from_lengths(mel_lens_video, max_mel_len_video)
        # print("Mels: ", mels.shape)
        # print("Mel mask: ", mel_masks.shape)
        

        # (
        #     output,
        #     p_predictions,
        #     e_predictions,
        #     # log_d_predictions,
        #     # d_rounded,
        #     # mel_lens,
        #     mel_masks,
        # ) = self.variance_adaptor(
        #     output,
        #     visual_masks,
        #     mel_masks,
        #     max_mel_len,
        #     p_targets,
        #     e_targets,
        #     d_targets,
        #     p_control,
        #     e_control,
        #     d_control,
        # )

        output, mel_masks = self.decoder(output, mel_mask_video)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            # p_predictions,
            # e_predictions,
            # log_d_predictions,
            # d_rounded,
            text_masks,
            mel_mask_video,
            text_lens,
            mel_lens_video,
        )


class FastSpeech2Mel(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Mel, self).__init__()
        self.model_config = model_config

        self.text_encoder = TextEncoder(model_config)
        self.visual_encoder = VisualEncoder(model_config)
        self.text_video_aligner = ScaledDotProductAttention(model_config)
        self.scale_factor = model_config["transformer"]["upsample_scale_mel"]
        self.upsampler = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        # self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

    
    def forward(
        self,
        speakers,
        texts,
        text_lens,
        max_text_len,
        visual_emb,
        video_lens,
        max_video_len,
        mel_lens_video,
        max_mel_len_video,
        speaker_emb,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        frames=None
    ):

        # print("Visual emb: ", visual_emb.shape)
        # print("Speaker emb: ", speaker_emb.shape)

        text_masks = get_mask_from_lengths(text_lens, max_text_len)
        # print("Text mask: ", text_masks.shape)

        text_output = self.text_encoder(texts, text_masks)
        # print("Text encoder output: ", text_output.shape)

        visual_masks = get_mask_from_lengths(video_lens, max_video_len)
        # print("Video mask: ", visual_masks.shape)

        visual_output = self.visual_encoder(visual_emb, visual_masks)
        # print("Visual encoder output: ", visual_output.shape)

        output, attn_wts = self.text_video_aligner(visual_output, text_output, text_output)
        # print("Attn Output: ", output.shape)
        # print("Attn weights: ", attn_wts.shape)

        h_context = output + visual_output
        # print("H context: ", h_context.shape)
        h_mel = self.upsampler(h_context.permute(0,2,1)).permute(0,2,1)
        # print("H mels: ", h_mel.shape)

        repeat_len = h_mel.size(1)
        output = h_mel + speaker_emb.unsqueeze(1).expand(-1, repeat_len, -1)
        # print("Input to variance adaptor: ", output.shape)
        # print("Max mel len: ", max_mel_len)


        mel_mask_video = get_mask_from_lengths(mel_lens_video, max_mel_len_video)
        # print("Mels: ", mels.shape)
        # print("Mel mask: ", mel_masks.shape)
        

        # (
        #     output,
        #     p_predictions,
        #     e_predictions,
        #     # log_d_predictions,
        #     # d_rounded,
        #     # mel_lens,
        #     mel_masks,
        # ) = self.variance_adaptor(
        #     output,
        #     visual_masks,
        #     mel_masks,
        #     max_mel_len,
        #     p_targets,
        #     e_targets,
        #     d_targets,
        #     p_control,
        #     e_control,
        #     d_control,
        # )

        output, mel_masks = self.decoder(output, mel_mask_video)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            text_masks,
            mel_mask_video,
            text_lens,
            mel_lens_video,
        )


class FastSpeech2All(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2All, self).__init__()
        self.model_config = model_config
        self.text_encoder = TextEncoder(model_config)
        self.visual_encoder = VisualEncoder(model_config)
        self.text_video_aligner = ScaledDotProductAttention(model_config)
        self.scale_factor = model_config["transformer"]["upsample_scale_mel"]
        self.upsampler = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

    
    def forward(
        self,
        texts,
        text_lens,
        max_text_len,
        visual_emb,
        video_lens,
        max_video_len,
        mel_lens_video,
        max_mel_len_video,
        speaker_emb,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        frames=None
    ):

        # print("Visual emb: ", visual_emb.shape)
        # print("Speaker emb: ", speaker_emb.shape)

        text_masks = get_mask_from_lengths(text_lens, max_text_len)
        # print("Text mask: ", text_masks.shape)

        text_output = self.text_encoder(texts, text_masks)
        # print("Text encoder output: ", text_output.shape)

        visual_masks = get_mask_from_lengths(video_lens, max_video_len)
        # print("Video mask: ", visual_masks.shape)

        visual_output = self.visual_encoder(visual_emb, visual_masks)
        # print("Visual encoder output: ", visual_output.shape)

        output, attn_wts = self.text_video_aligner(visual_output, text_output, text_output)
        # print("Attn Output: ", output.shape)
        # print("Attn weights: ", attn_wts.shape)

        h_context = output + visual_output
        # print("H context: ", h_context.shape)
        h_mel = self.upsampler(h_context.permute(0,2,1)).permute(0,2,1)
        # print("H mels: ", h_mel.shape)

        repeat_len = h_mel.size(1)
        output = h_mel + speaker_emb.unsqueeze(1).expand(-1, repeat_len, -1)
        # print("Input to variance adaptor: ", output.shape)
        # print("Max mel len: ", max_mel_len)


        mel_mask_video = get_mask_from_lengths(mel_lens_video, max_mel_len_video)
        # print("Mels: ", mels.shape)
        # print("Mel mask: ", mel_masks.shape)
        

        output, mel_masks = self.decoder(output, mel_mask_video)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            text_masks,
            mel_mask_video,
            text_lens,
            mel_lens_video,
            attn_wts
        )


class Conv3d(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv3d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm3d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            try:
                out += x
            except:
                print("Residual conv error")
        return self.act(out)

class FastSpeech2AllFace(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2AllFace, self).__init__()
        self.model_config = model_config
        self.text_encoder = TextEncoder(model_config)
        self.visual_encoder = VisualEncoder(model_config)
        self.text_video_aligner = ScaledDotProductAttention(model_config)
        self.scale_factor = model_config["transformer"]["upsample_scale_mel"]
        self.upsampler = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.face_encoder = nn.Sequential(
            Conv3d(3, 32, kernel_size=5, stride=(1,2,2), padding=2),             # Bx32x25x80x80
            Conv3d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),  

            Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1)),            # Bx64x25x40x40
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1)),           # Bx128x25x20x20
            # Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1)),          # Bx256x25x10x10
            # Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1)),          # Bx512x25x5x5
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            
            Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1)),          # Bx512x25x3x3
            # Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1,3,3), padding=(0, 1, 1)),          # Bx512x25x3x3
            Conv3d(512, 512, kernel_size=1, stride=1, padding=0)
        )

    
    def forward(
        self,
        texts,
        text_lens,
        max_text_len,
        face_sequence,
        video_lens,
        max_video_len,
        mel_lens_video,
        max_mel_len_video,
        speaker_emb,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        frames=None
    ):

        # print("Input frames: ", face_sequence.shape)
        # print("Speaker emb: ", speaker_emb.shape)

        face_sequence_permuted = face_sequence.permute(0,4,1,2,3)       
        # print("Face input permuted: ", face_sequence_permuted.size())     # Bx3x25x96x96

        face_enc = self.face_encoder(face_sequence_permuted)
        # print("Face encoding output: ", face_enc.size())                  # Bx512x25x1x1

        face_enc = face_enc.view(-1, face_enc.size(1), face_enc.size(2))
        # print("Face enc reshaped: ", face_enc.size())                     # Bx512x25

        # face_output = self.time_upsampler(face_enc)
        visual_emb = face_enc.permute(0, 2, 1)                  
        # print("Face output: ", visual_emb.size())                        # Bx100x512

        text_masks = get_mask_from_lengths(text_lens, max_text_len)
        # print("Text mask: ", text_masks.shape)

        text_output = self.text_encoder(texts, text_masks)
        # print("Text encoder output: ", text_output.shape)

        visual_masks = get_mask_from_lengths(video_lens, max_video_len)
        # print("Video mask: ", visual_masks.shape)

        visual_output = self.visual_encoder(visual_emb, visual_masks)
        # print("Visual encoder output: ", visual_output.shape)

        output, attn_wts = self.text_video_aligner(visual_output, text_output, text_output)
        # print("Attn Output: ", output.shape)
        # print("Attn weights: ", attn_wts.shape)

        h_context = output + visual_output
        # print("H context: ", h_context.shape)
        h_mel = self.upsampler(h_context.permute(0,2,1)).permute(0,2,1)
        # print("H mels: ", h_mel.shape)

        repeat_len = h_mel.size(1)
        output = h_mel + speaker_emb.unsqueeze(1).expand(-1, repeat_len, -1)
        # print("Input to variance adaptor: ", output.shape)
        # print("Max mel len: ", max_mel_len)


        mel_mask_video = get_mask_from_lengths(mel_lens_video, max_mel_len_video)
        # print("Mels: ", mels.shape)
        # print("Mel mask: ", mel_masks.shape)
        

        output, mel_masks = self.decoder(output, mel_mask_video)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            text_masks,
            mel_mask_video,
            text_lens,
            mel_lens_video,
        )