import torch
import torch.nn as nn
from Self_attention import TriangularCausalMask,FullAttention,AttentionLayer
from encoder import Encoder, EncoderLayer

class TA(nn.Module):
    def __init__(self,c_in,c_out,dropout):
        super(TA, self).__init__()
        self.d_encoder=c_in
        self.encoder=Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(FullAttention(False, attention_dropout=dropout), self.d_encoder, n_heads=8),
                            self.d_encoder,
                            self.d_encoder*2,
                            dropout=dropout,
                            activation='relu',
                            flag="TA"
                        ) for l in range(2)
                    ],#encoder层的列表
                    None,
                    norm_layer=torch.nn.LayerNorm(self.d_encoder)
                )

    def forward(self,x):
        B,T,E,H,W = x.shape
        x = x.permute(0,3,4,1,2).reshape(B*H*W,T,E)
        x = self.encoder(x)
        x = x.reshape(B,H,W,T,E).permute(0,3,4,1,2)
        return x
