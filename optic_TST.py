import torch.nn as nn
from model import TST


class Optic_TST(nn.Module):
    """Only optic segmentation network."""

    def __init__(self, T=[23,23], input_c=[10,10], image_size=128, label_c=9, leve_dims=[32,64,128], dropout=0.5, dim_weight=64):
        super(Optic_TST, self).__init__()

        self.T=T
        self.input_c=input_c
        self.dims=leve_dims

        self.first_optic_conv_0=nn.Conv2d(input_c[0], int(self.dims[0]/2), kernel_size=3, padding=1)#此层前后将B与T叠加和拆分
        self.first_optic_bn_0 = nn.BatchNorm2d(int(self.dims[0]/2))
        self.first_optic_Relu_0 = nn.ReLU()
        self.first_optic_conv_1=nn.Conv2d(int(self.dims[0]/2), self.dims[0], kernel_size=3, padding=1)#此层前后将B与T叠加和拆分
        self.first_optic_bn_1 = nn.BatchNorm2d(self.dims[0])
        self.first_optic_Relu_1 = nn.ReLU()
        self.first_optic_globalavg = nn.AvgPool2d(kernel_size=image_size)
        self.second_optic_conv=nn.Conv2d(self.dims[0], self.dims[0], kernel_size=3, padding=1)
        self.second_optic_bn = nn.BatchNorm2d(self.dims[0])
        self.second_optic_Relu = nn.ReLU()
        self.second_optic_globalavg = nn.AvgPool2d(kernel_size=image_size)

        self.Optic =TST(T=T[0], input_c=self.dims[0], image_size=image_size, label_c=label_c, leve_dims=leve_dims, dropout=dropout, dim_weight=dim_weight)

        self.classconv_optic = nn.Conv2d(dim_weight, label_c, kernel_size=3, padding=1)

            
    def forward(self, x1, x2):

        B, T1, E1, H, W=x1.shape
        x1 = x1.reshape(B*T1,E1,H,W)

        x1 = self.first_optic_conv_0(x1)
        x1 = self.first_optic_bn_0(x1)
        x1 = self.first_optic_Relu_0(x1)

        x1 = self.first_optic_conv_1(x1)
        x1 = self.first_optic_bn_1(x1)
        x1 = self.first_optic_Relu_1(x1)


        x1_TST = x1.reshape(B,T1,self.dims[0],H,W)

        x1_TST = self.Optic(x1_TST)
        output_optic = self.classconv_optic(x1_TST)


        return output_optic
