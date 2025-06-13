import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import GPT2ForSequenceClassification, LlamaModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from layers.Embed import DataEmbedding
from peft import LoraModel, LoraConfig, get_peft_model, get_peft_config

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Res_block(nn.Module):
    def __init__(self, in_planes):
        super(Res_block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.device = torch.device('cuda:{}'.format(config.gpu))
        self.mlp = config.mlp
        self.res_layers = config.res_layers
        self.pred_len = config.pred_len
        self.prev_len = config.prev_len
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.d_ff = config.llm_d_ff
        self.d_model = config.llm_d_model

        self.K = config.K
        self.UQh = config.UQh
        self.UQv = config.UQv
        self.BQh = config.BQh
        self.BQv = config.BQv
        self.Nt = config.UQh * config.UQv
        self.Nr = config.BQh * config.BQv
        self.mul = config.prev_len * config.K * config.UQh * config.UQv * config.BQh * config.BQv
        self.enc_in = config.K * config.UQh * config.UQv * config.BQh * config.BQv
        self.c_out = config.K * config.UQh * config.UQv * config.BQh * config.BQv

        if self.config.data == 'QuaDriGa':
            self.enc_embedding1 = DataEmbedding(2 * self.enc_in, self.d_model, config.dropout_gpt)
        elif self.config.data == 'SJTU':
            self.enc_embedding1 = DataEmbedding(2 * self.config.subcarrier_num, self.d_model, config.dropout_gpt)
        if config.pretrain:
            if config.llm_type == 'gpt2-medium':
                self.llm = GPT2Model.from_pretrained('/workspace/LLMs/gpt2-medium', output_attentions=True, output_hidden_states=True, )
                self.llm.h = self.llm.h[:config.llm_layers]
                self.llm_dim = 1024
            elif config.llm_type == 'gpt2-large':
                self.llm = GPT2Model.from_pretrained('/workspace/LLMs/gpt2-large', output_attentions=True, output_hidden_states=True, )
                self.llm.h = self.llm.h[:config.llm_layers]
                self.llm_dim = 1280
            elif config.llm_type == 'gpt2-xl':
                self.llm = GPT2Model.from_pretrained('/workspace/LLMs/gpt2-xl', output_attentions=True, output_hidden_states=True, )
                self.llm.h = self.llm.h[:config.llm_layers]
                self.llm_dim = 1600
            elif config.llm_type == 'gpt2':
                self.llm = GPT2Model.from_pretrained('/workspace/LLMs/gpt2', output_attentions=True, output_hidden_states=True, )
                self.llm.h = self.llm.h[:config.llm_layers]
                self.llm_dim = 768
            elif config.llm_type == 'llama':
                self.llama_config = LlamaConfig.from_pretrained('/workspace/LLMs/llama-7b')
                self.llama_config.num_hidden_layers = 1
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True
                self.llama_config.use_cache = False
                self.llm = LlamaModel.from_pretrained(
                    '/workspace/LLMs/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    torch_dtype=torch.float16,
                    # device_map="auto",  # 自动分配多卡
                    # load_in_4bit=True
                )
                self.llm.gradient_checkpointing_enable()
                self.llm.enable_input_require_grads()
                self.llm.is_parallelizable = True
                self.llm.model_parallel = True
                self.llm_dim = 4096
                # self.tokenizer = LlamaTokenizer.from_pretrained(
                #     '/workspace/LLMs/llama-7b',
                #     trust_remote_code=True,
                #     local_files_only=True
                # )

        if config.lora and config.llm_type == 'llama':
            lora_config=LoraConfig(
                # tasj_type="SEQ_2_SEQ_LM",
                r=8,
                lora_alpha=32,
                lora_dropout=0.01,
                target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
            )
            self.llm=get_peft_model(self.llm,lora_config)
            self.llm.print_trainable_parameters()
        else:
            # # 设置可学习的参数
            if config.freeze and config.pretrain:
                for i, (name, param) in enumerate(self.llm.named_parameters()):
                    if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                        param.requires_grad = True
                    elif 'mlp' in name and config.mlp == 1:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                    # if "attn" in name:
                    #     param.requires_grad = False
        if config.use_gpu:
            device = torch.device('cuda:{}'.format(config.gpu))
            self.llm.to(device=device)
        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.patch_layer_fre = nn.Linear(self.patch_size, self.patch_size)
        self.predict_linear_pre = nn.Linear(self.prev_len, self.prev_len)
        if self.config.data == 'QuaDriGa':
            self.out_layer_dim = nn.Linear(config.llm_d_ff, self.c_out * 2)
        elif self.config.data == 'SJTU':
            self.out_layer_dim = nn.Linear(config.llm_d_ff, self.config.subcarrier_num * 2)

        self.output_layer_time = nn.Sequential( 
            nn.Linear(self.prev_len, self.pred_len)
        )

        self.RB_e = nn.Sequential(nn.Conv2d(2, config.res_dim, 3, 1, 1))
        self.RB_f = nn.Sequential(nn.Conv2d(2, config.res_dim, 3, 1, 1))
        for i in range(self.res_layers):
            self.RB_e.append(Res_block(config.res_dim))
            self.RB_f.append(Res_block(config.res_dim))
        self.RB_e.append(nn.Conv2d(config.res_dim, 2, 3, 1, 1))
        self.RB_f.append(nn.Conv2d(config.res_dim, 2, 3, 1, 1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        B, L, enc_in = x_enc.shape  # [B, L, D]
        # process in delay domain
        x_enc_r = rearrange(x_enc, 'b l (k o) -> b l k o', o=2)
        x_enc_complex = torch.complex(x_enc_r[:, :, :, 0], x_enc_r[:, :, :, 1])
        x_enc_delay = torch.fft.ifft(x_enc_complex, dim=2)
        x_enc_delay = torch.cat([torch.real(x_enc_delay), torch.imag(x_enc_delay)], dim=2)
        x_enc_delay = x_enc_delay.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc_delay = self.patch_layer(x_enc_delay.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_delay = x_enc_delay.reshape(B, L, enc_in)
        x_enc_delay = rearrange(x_enc_delay, 'b l (k o) -> b o l k', o=2)
        x_enc_delay = self.RB_f(x_enc_delay)
        # process in frequency domain
        x_enc_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc_fre = self.patch_layer(x_enc_fre.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_fre = x_enc_fre.reshape(B, L, enc_in)
        x_enc_fre = rearrange(x_enc_fre, 'b l (k o) -> b o l k', o=2)
        x_enc_fre = self.RB_e(x_enc_fre)

        x_enc = x_enc_fre + x_enc_delay
        x_enc = rearrange(x_enc, 'b o l k -> b l (k o)', o=2)  # [B, L, D]

        enc_out = self.enc_embedding1(x_enc, x_mark_enc)  # [B, L, 768]

        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        enc_out = torch.nn.functional.pad(enc_out, (0, self.llm_dim - enc_out.shape[-1]))

        dec_out = self.llm(inputs_embeds=enc_out).last_hidden_state  # [B, L, 768]
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = self.out_layer_dim(dec_out)
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)

        dec_out = dec_out * std + mean

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

if __name__ == '__main__':
    import torch

    device = torch.device('cuda')
    model = Model(UQh=1, UQv=1, BQh=1, BQv=1).to(device)
    inputs = torch.rand(3, 16, 96).to(device)
    out = model(inputs, None, None, None)
    print(out.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
