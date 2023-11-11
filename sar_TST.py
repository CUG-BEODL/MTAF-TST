import torch.nn as nn
from model import TST


class Sar_TST(nn.Module):
    """Only sar segmentation network."""

    def __init__(self, T=[23,23], input_c=[10,10], image_size=128, label_c=9, leve_dims=[32,64,128], dropout=0.5, dim_weight=64):
        super(Sar_TST, self).__init__()

        self.T=T
        self.input_c=input_c
        self.dims=leve_dims

        self.first_sar_conv_0=nn.Conv2d(input_c[1], int(self.dims[0]/2), kernel_size=3, padding=1)
        self.first_sar_bn_0 = nn.BatchNorm2d(int(self.dims[0]/2))
        self.first_sar_Relu_0 = nn.ReLU()
        self.first_sar_conv_1=nn.Conv2d(int(self.dims[0]/2), self.dims[0], kernel_size=3, padding=1)
        self.first_sar_bn_1 = nn.BatchNorm2d(self.dims[0])
        self.first_sar_Relu_1 = nn.ReLU()
        self.first_sar_globalavg = nn.AvgPool2d(kernel_size=image_size)
        self.second_sar_conv=nn.Conv2d(self.dims[0], self.dims[0], kernel_size=3, padding=1)
        self.second_sar_bn = nn.BatchNorm2d(self.dims[0])
        self.second_sar_Relu = nn.ReLU()
        self.second_sar_globalavg = nn.AvgPool2d(kernel_size=image_size)

        self.Sar =TST(T=T[1], input_c=self.dims[0], image_size=image_size, label_c=label_c, leve_dims=leve_dims, dropout=dropout, dim_weight=dim_weight)

        self.classconv_sar = nn.Conv2d(dim_weight, label_c, kernel_size=3, padding=1)
        
            
    def forward(self,x1, x2):

        B, T2, E2, H, W=x2.shape

        x2 = x2.reshape(B*T2,E2,H,W)

        x2 = self.first_sar_conv_0(x2)
        x2 = self.first_sar_bn_0(x2)
        x2 = self.first_sar_Relu_0(x2)

        x2 = self.first_sar_conv_1(x2)
        x2 = self.first_sar_bn_1(x2)
        x2 = self.first_sar_Relu_1(x2)


        x2_TST = x2.reshape(B,T2,self.dims[0],H,W)
        x2_TST = self.Sar(x2_TST)
        output_sar = self.classconv_sar(x2_TST)

        return output_sar
