import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Self_attention import TriangularCausalMask,FullAttention,AttentionLayer
from encoder import Encoder, EncoderLayer

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w

class SA(nn.Module):
    def __init__(self,img_size,in_dims,out_dims,dropout):
        super(SA, self).__init__()
        self.in_dims=in_dims
        self.d_encoder=out_dims
        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=3,
            stride=2,
            in_chans=self.in_dims,
            embed_dim=self.d_encoder,
        )
        self.encoder=Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(FullAttention(False, attention_dropout=dropout), self.d_encoder, n_heads=8),
                            self.d_encoder,
                            self.d_encoder*2,
                            dropout=dropout,
                            activation='gelu',
                            flag="SA"
                        ) for l in range(2)
                    ],#encoder层的列表
                    None,
                    norm_layer=torch.nn.LayerNorm(self.d_encoder)
                )

    def forward(self,x):
        B,T,E,H,W = x.shape
        x = x.reshape(-1,x.shape[2],x.shape[3],x.shape[4])
        x, h, w = self.patch_embed(x)
        x = self.encoder(x, H=h, W=w)
        x = x.reshape(B,T,h,w,self.d_encoder).permute(0,1,4,2,3)
        return x
