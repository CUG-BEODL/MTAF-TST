import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class PositionalEncoding(nn.Module): 
    "Implement the PE function." 
    def __init__(self, d_model, dropout, max_len=5000): 
        #d_model=512,dropout=0.1,
        #max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，
        #一般100或者200足够了。
        super(PositionalEncoding, self).__init__() 
        self.dropout = nn.Dropout(p=dropout) 

        # Compute the positional encodings once in log space. 
        pe = torch.zeros(max_len, d_model) 
        #(5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
        #每个位置用一个512维度向量来表示其位置编码
        position = torch.arange(0, max_len).unsqueeze(1) 
        # (5000) -> (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
            -(math.log(10000.0) / d_model)) 
            # (0,2,…, 4998)一共准备2500个值，供sin, cos调用
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数下标的位置
        pe = pe.unsqueeze(0) 
        # (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
        self.register_buffer('pe', pe) 
    def forward(self, x): 
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
        # 接受1.Embeddings的词嵌入结果x，
        #然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
        #例如，假设x是(30,10,512)的一个tensor，
        #30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
        #则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
        #在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
        #保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
        return self.dropout(x) # 增加一次dropout操作
        # 注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数。

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).reshape(B, N, C)
        return x
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", flag="TA"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.flag = flag
        self.PE = PositionalEncoding(d_model,0.0)
        self.pre_norm = nn.LayerNorm(d_model)
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        if(self.flag == "SA"):
            self.dwconv = DWConv(d_ff)
        self.dropout = nn.Dropout(dropout) 
        self.activation = F.relu if activation == "relu" else F.gelu
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, H=None, W=None, attn_mask=None):
        # x [B, T, E]
        x = self.PE(x)
        y = self.pre_norm(x)
        y = x + self.dropout(self.attention(
            y, y, y,
            attn_mask = attn_mask
        ))
        x = y
        y = self.norm1(y)
        y = self.linear1(y)
        if(self.flag == "SA"):
            y = self.dwconv(y, H, W)
        y = self.dropout(self.activation(y))
        y = self.dropout(self.linear2(y))
        return self.norm2(x+y)

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, H=None, W=None, attn_mask=None):
        # x [B, L, D]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, H, W, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, H, W, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x