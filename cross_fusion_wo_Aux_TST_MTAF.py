import torch.nn as nn

class Cross_Wo_Aux_TST_MTAF(nn.Module):
    """Twins_Add_Cross 移除辅助分类器、TST、MTAF"""

    def __init__(self, T=[23,23], input_c=[10,10], image_size=128, label_c=9, leve_dims=[32,64,128], dropout=0.5, dim_weight=64):
        super(Cross_Wo_Aux_TST_MTAF, self).__init__()

        self.T=T
        self.input_c=input_c
        self.dims=leve_dims

        self.first_optic_conv_0=nn.Conv2d(input_c[0], int(self.dims[0]/2), kernel_size=3, padding=1)
        self.first_optic_bn_0 = nn.BatchNorm2d(int(self.dims[0]/2))
        self.first_optic_Relu_0 = nn.ReLU()
        self.first_optic_conv_1=nn.Conv2d(int(self.dims[0]/2), self.dims[0], kernel_size=3, padding=1)
        self.first_optic_bn_1 = nn.BatchNorm2d(self.dims[0])
        self.first_optic_Relu_1 = nn.ReLU()

        self.first_sar_conv_0=nn.Conv2d(input_c[1], int(self.dims[0]/2), kernel_size=3, padding=1)
        self.first_sar_bn_0 = nn.BatchNorm2d(int(self.dims[0]/2))
        self.first_sar_Relu_0 = nn.ReLU()
        self.first_sar_conv_1=nn.Conv2d(int(self.dims[0]/2), self.dims[0], kernel_size=3, padding=1)
        self.first_sar_bn_1 = nn.BatchNorm2d(self.dims[0])
        self.first_sar_Relu_1 = nn.ReLU()
        self.Optic_Avg=nn.AvgPool3d(kernel_size=(T[0],1,1))
        self.Sar_Avg=nn.AvgPool3d(kernel_size=(T[1],1,1))


        self.Conconv=nn.Conv2d(self.dims[0], dim_weight, kernel_size=3, padding=1)
        self.Conbn = nn.BatchNorm2d(dim_weight)
        self.ConRelu = nn.ReLU()
        self.classconv = nn.Conv2d(dim_weight, label_c, kernel_size=3, padding=1)

        
            
    def forward(self, x1, x2):

        B, T1, E1, H, W=x1.shape
        x1 = x1.reshape(B*T1,E1,H,W)

        x1 = self.first_optic_conv_0(x1)
        x1 = self.first_optic_bn_0(x1)
        x1 = self.first_optic_Relu_0(x1)

        x1 = self.first_optic_conv_1(x1)
        x1 = self.first_optic_bn_1(x1)
        x1 = self.first_optic_Relu_1(x1)

        B, T2, E2, H, W=x2.shape

        x2 = x2.reshape(B*T2,E2,H,W)

        x2 = self.first_sar_conv_0(x2)
        x2 = self.first_sar_bn_0(x2)
        x2 = self.first_sar_Relu_0(x2)

        x2 = self.first_sar_conv_1(x2)
        x2 = self.first_sar_bn_1(x2)
        x2 = self.first_sar_Relu_1(x2)

        x1 = x1.reshape(B, T1, -1, H, W)
        x1 = x1.permute(0,2,1,3,4)
        x1 = self.Optic_Avg(x1)
        x1 = x1.squeeze()
        
        x2 = x2.reshape(B, T2, -1, H, W)
        x2 = x2.permute(0,2,1,3,4)
        x2 = self.Sar_Avg(x2)
        x2 = x2.squeeze()
        
        
        x_con = self.Conconv(x1+x2)
        x_con = self.Conbn(x_con)
        x_con = self.ConRelu(x_con)
        
        output = self.classconv(x_con)

        
        return output
