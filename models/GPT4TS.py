import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel, GPT2Model, LlamaModel
from einops import rearrange
# from layers.Embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                # self.gpt2 = GPT2Model.from_pretrained('/workspace/LLMs/gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                self.gpt2 = LlamaModel.from_pretrained('/workspace/LLMs/llama-7b', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model\
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            # self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        # 如果使用预训练的模型和冻结部分参数
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs

if __name__ == '__main__':
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='GPT4TS')
    parser.add_argument('--is_gpt', type=int, default=1, help='Use GPT2 as backbone')
    parser.add_argument('--pretrain', type=int, default=1, help='Use pre-trained GPT2')
    parser.add_argument('--freeze', type=int, default=1, help='Freeze GPT2 parameters')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--stride', type=int, default=8, help='Stride for patching')
    parser.add_argument('--gpt_dim', type=int, default=768, help='GPT dimension')
    parser.add_argument('--gpt_layers', type=int, default=12, help='Number of GPT layers')
    parser.add_argument('--d_model', type=int, default=4096, help='Model dimension')
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length')
    parser.add_argument('--pred_len', type=int, default=4, help='Prediction length')

    args = parser.parse_args()

    device = torch.device('cpu')
    model = GPT4TS(args, device).to(device)
    inputs = torch.rand(3, 16, 96).to(device)
    out = model(inputs, None)
    print(out.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
