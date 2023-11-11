import torch.nn as nn
from TA import TA
from SA import SA

class TST(nn.Module):
    """TST segmentation network."""

    def __init__(self, T=23, input_c=10, image_size=128, label_c=9, leve_dims=[32,64,128], dropout=0.5, dim_weight=64):
        super(TST, self).__init__()
        
        self.dims=leve_dims
        self.levels=len(leve_dims)

        self.starconv=nn.Conv2d(input_c, self.dims[0], kernel_size=3, padding=1)
        self.starbn = nn.BatchNorm2d(self.dims[0])
        self.starRelu = nn.ReLU()
        
        self.TAs=nn.ModuleList()
        self.SAs=nn.ModuleList()
        self.ConvTran=nn.ModuleList()
        
        
        for i in range(self.levels-1):
            self.TAs.append(TA(self.dims[i], self.dims[i], dropout))
            self.SAs.append(SA(image_size//(2**i),self.dims[i], self.dims[i+1], dropout))
            self.ConvTran.append(nn.ConvTranspose2d(self.dims[self.levels-i-1], self.dims[self.levels-i-2], kernel_size=3, stride=2, padding=1, output_padding=1))

        self.Conv3d=nn.Conv3d(self.dims[0], self.dims[0]*4, kernel_size=(T,1,1))

        self.conv_one=nn.Conv2d(self.dims[0]*4, self.dims[0]*2, kernel_size=3, padding=1)
        self.bn_one = nn.BatchNorm2d(self.dims[0]*2)
        self.Relu_one = nn.ReLU()#inplace=True
        self.shortcut_one = nn.Conv2d(self.dims[0]*4, self.dims[0]*2, kernel_size=1)

        self.conv_two=nn.Conv2d(self.dims[0]*2, dim_weight, kernel_size=3, padding=1)
        self.bn_two = nn.BatchNorm2d(dim_weight)
        self.Relu_two = nn.ReLU()
        self.shortcut_two = nn.Conv2d(self.dims[0]*2, dim_weight, kernel_size=1)
        
            
    def forward(self, x):
        #x:B, T, E, H, W
        #print(x.shape)
        B, T, E, H, W=x.shape
        x = x.reshape(B*T, E, H, W)
        x = self.starconv(x)#32 9 50 50
        x = self.starbn(x)#C32 32 50 50
        x = self.starRelu(x)
        x = x.reshape(B, T, self.dims[0], H, W)#16 2 32 50 50
        DwFeatures_level=[]
        DwFeatures_level.append(x)
        for i in range(self.levels-1):
            x = self.TAs[i](x)#16 2 32 50 50
            x = self.SAs[i](x)#16 2 64 25 25
            DwFeatures_level.append(x)#B*T, self.dims[i], h, w
        b, t, e, h, w = DwFeatures_level[self.levels-1].shape
        x_up = self.ConvTran[0](DwFeatures_level[self.levels-1].reshape(b*t,-1,h,w))
        b, t, e, h, w = DwFeatures_level[self.levels-2].shape
        x_up = x_up+DwFeatures_level[self.levels-2].reshape(b*t,-1,h,w)#B*T, self.dims[0], H, W
        for i in range(1,self.levels-1):
            x_up = self.ConvTran[i](x_up)

            b_t, e, h, w = x_up.shape
            x_up = x_up+DwFeatures_level[self.levels-2-i].reshape(b_t,-1,h,w)#B*T, self.dims[0], H, W

        x_up = x_up.reshape(B, T, -1, H, W)
        x_up = x_up.permute(0,2,1,3,4)
        x_up = self.Conv3d(x_up)
        x_up = x_up.squeeze()
        x = self.conv_one(x_up)
        x = self.bn_one(x)
        x = self.Relu_one(x)
        x = x+self.shortcut_one(x_up)

        y = x
        x = self.conv_two(x)
        x = self.bn_two(x)
        x = self.Relu_two(x)
        x = x+self.shortcut_two(y)
        
        x=x_up

        
        return x
