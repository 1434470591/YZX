import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import DataEmbedding
from layers.Encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from layers.Decoder import Decoder, DecoderLayer
from layers.Attention import FullAttention, ProbAttention, AttentionLayer

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pred_len = config.pred_len
        self.attn = config.attn
        self.output_attention = config.output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(config.enc_in, config.d_model, config.dropout_trans)
        self.dec_embedding = DataEmbedding(config.dec_in, config.d_model, config.dropout_trans)
        # Attention
        Attn = ProbAttention if config.attn == 'prob' else FullAttention
       
        
        if config.stack:
            stacks = list(range(config.e_layers, 2, -1))  # you can customize here  # [4 3]
            encoders = [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                Attn(False, config.factor, attention_dropout=config.dropout_trans, output_attention=config.output_attention),
                                config.d_model, config.n_heads),
                            config.d_model,
                            config.d_ff,
                            dropout=config.dropout_trans,
                            activation=config.activation
                        ) for l in range(el)
                    ],
                    [
                        ConvLayer(
                            config.d_model
                        ) for l in range(el - 1)
                    ] if config.distil else None,
                    norm_layer=torch.nn.LayerNorm(config.d_model)  # 层正则化
                ) for el in stacks]
            self.encoder = EncoderStack(encoders)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, config.factor, attention_dropout=config.dropout_trans, output_attention=config.output_attention),
                                   config.d_model, config.n_heads),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout_trans,
                    activation=config.activation
                ) for l in range(config.e_layers)
            ],
            [
                ConvLayer(
                    config.d_model
                ) for l in range(config.e_layers - 1)
            ] if config.distil else None,
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, config.factor, attention_dropout=config.dropout_trans, output_attention=False),
                                   config.d_model, config.n_heads),
                    AttentionLayer(FullAttention(False, config.factor, attention_dropout=config.dropout_trans, output_attention=False),
                                   config.d_model, config.n_heads),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout_trans,
                    activation=config.activation,
                )
                for l in range(config.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(config.d_model, config.c_out, bias=True)  # 线性层只改变最后一个维度

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]