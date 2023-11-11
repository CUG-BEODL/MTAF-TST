from osgeo import gdal
import numpy as np
import os
import pandas as pd
import random
import torchvision.transforms as transforms
import torch

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
        return
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_bands = dataset.RasterCount #波段数
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
    im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
    im_proj = dataset.GetProjection()#获取投影信息
    # im_blueBand =  im_data[0,0:im_height,0:im_width]#获取蓝波段
    # im_greenBand = im_data[1,0:im_height,0:im_width]#获取绿波段
    # im_redBand =   im_data[2,0:im_height,0:im_width]#获取红波段
    # im_nirBand = im_data[3,0:im_height,0:im_width]#获取近红外波段
    return im_data,im_width,im_height,im_bands,im_geotrans,im_proj

def writeTiff(im_data,im_width,im_height,im_bands,im_geotrans,im_proj,path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape
        #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset
#语义分割裁剪
def wuhan_image2patch():
    im_data_p,im_width,im_height,im_bands_p,im_geotrans_p,im_proj_p=readTif("Optic\wuhan.tiff")
    im_data_s,im_width,im_height,im_bands_s,im_geotrans_s,im_proj_s=readTif("SAR\wuhan.tiff")
    im_data_l,im_width,im_height,im_bands_l,im_geotrans_l,im_proj_l=readTif("Label\wuhan.tiff")
    print(im_data_p.shape,im_data_s.shape,im_data_l.shape)
    window=64
    path="wuhan-Patch64"
    for i in np.arange(0,im_data_l.shape[0],window):
        for j in np.arange(0,im_data_l.shape[1],window):
            if((i+window)<im_data_l.shape[0] and (j+window)<im_data_l.shape[1]):
                print(os.path.join(path,"P",str(i)+"_"+str(j)+".tiff"))
                cilp_p=im_data_p[:,i:i+window,j:j+window]
                cilp_s=im_data_s[:,i:i+window,j:j+window]
                cilp_l=im_data_l[i:i+window,j:j+window]
                print(cilp_p.shape)
                im_geotrans_p_n = (im_geotrans_p[0]+j*(10.0),10.0, 0.0,im_geotrans_p[3]+i*(-10.0),0.0, -10.0)
                writeTiff(cilp_p,window,window,im_bands_p,im_geotrans_p_n,im_proj_p,os.path.join(path,"P",str(i)+"_"+str(j)+".tiff"))#
                writeTiff(cilp_s,window,window,im_bands_s,im_geotrans_p_n,im_proj_s,os.path.join(path,"S",str(i)+"_"+str(j)+".tiff"))#
                writeTiff(cilp_l,window,window,im_bands_l,im_geotrans_p_n,im_proj_l,os.path.join(path,"gt",str(i)+"_"+str(j)+".tiff"))#
# wuhan_image2patch()

def image2patch(area):
    im_data_p,im_width,im_height,im_bands_p,im_geotrans_p,im_proj_p=readTif("Optic\\area"+area+".tiff")
    im_data_s,im_width,im_height,im_bands_s,im_geotrans_s,im_proj_s=readTif("SAR\\area"+area+".tiff")
    im_data_l,im_width,im_height,im_bands_l,im_geotrans_l,im_proj_l=readTif("Label\\area"+area+".tiff")
    print(im_data_p.shape,im_data_s.shape,im_data_l.shape)
    window=16
    path="Ili-Patch16"
    for i in np.arange(0,im_data_l.shape[0],window):
        for j in np.arange(0,im_data_l.shape[1],window):
            if((i+window)<im_data_l.shape[0] and (j+window)<im_data_l.shape[1]):
                print(os.path.join(path,"P",area+"_"+str(i)+"_"+str(j)+".tiff"))
                cilp_p=im_data_p[:,i:i+window,j:j+window]
                cilp_s=im_data_s[:,i:i+window,j:j+window]
                cilp_l=im_data_l[i:i+window,j:j+window]
                print(cilp_p.shape)
                im_geotrans_p_n = (im_geotrans_p[0]+j*(10.0),10.0, 0.0,im_geotrans_p[3]+i*(-10.0),0.0, -10.0)
                writeTiff(cilp_p,window,window,im_bands_p,im_geotrans_p_n,im_proj_p,os.path.join(path,"P",area+"_"+str(i)+"_"+str(j)+".tiff"))#
                writeTiff(cilp_s,window,window,im_bands_s,im_geotrans_p_n,im_proj_s,os.path.join(path,"S",area+"_"+str(i)+"_"+str(j)+".tiff"))#
                writeTiff(cilp_l,window,window,im_bands_l,im_geotrans_p_n,im_proj_l,os.path.join(path,"gt",area+"_"+str(i)+"_"+str(j)+".tiff"))#
# image2patch("5")

def get_mean_std(path="Ili-Patch16\P",T=40,band=4):
    total_data=[]
    mean=[]
    std=[]
    for filename in os.listdir(path):
        file_path=os.path.join(path,filename)
        print(file_path)
        im_data_p,im_width,im_height,im_bands_p,im_geotrans_p,im_proj_p=readTif(file_path)
        print(im_data_p.shape)#T*band H W
        im_data_p=im_data_p.reshape(im_data_p.shape[0],-1)
        print(im_data_p.shape)#T*band H*W
        total_data.append(im_data_p)
    total_data=np.concatenate(total_data,axis=1)#T*band H*W*n
    print(total_data.shape)
    total_data=total_data.T#H*W*n T*band 
    total_data=total_data.reshape(total_data.shape[0],T,band);
    for i in range(band):
        mean.append(np.mean(total_data[:,:,i]))
        std.append(np.std(total_data[:,:,i]))
    return mean,std
# mean,std=get_mean_std()

def Norm(path="Ili-Patch16\S",T=46,band=2,win=16):
    mean,std=get_mean_std(path,T,band)
    output_path=path+"_norm"
    if(os.path.exists(output_path)==False):
        os.makedirs(output_path)
    for filename in os.listdir(path):
        file_path=os.path.join(path,filename)
        # print(file_path)
        im_data_p,im_width,im_height,im_bands_p,im_geotrans_p,im_proj_p=readTif(file_path)
        # print(im_data_p.shape)
        im_data_p=im_data_p.reshape(im_data_p.shape[0],-1).T
        # print(im_data_p.shape)
        im_data_p=im_data_p.reshape(im_data_p.shape[0],T,band).astype(np.float64)
        # print("\n")
        # print(im_data_p)
        new_patch=np.zeros_like(im_data_p)
        for i in range(band):
            new_patch[:,:,i]=(im_data_p[:,:,i]-mean[i])/std[i]
        # print(new_patch)
        # print("\n")
        new_patch=new_patch.reshape(new_patch.shape[0],T*band)
        # print(new_patch.shape)
        new_patch=new_patch.T.reshape(-1,win,win)
        # print(new_patch.shape)
        writeTiff(new_patch,im_width,im_height,im_bands_p,im_geotrans_p,im_proj_p,os.path.join(output_path,filename))
# Norm(path="Ili-Patch16\P",T=40,band=4,win=16)
# Norm(path="wuhan-Patch64\P",T=16,band=4,win=64)
# Norm(path="wuhan-Patch64\S",T=23,band=2,win=64)

def getTrainCsv(path="Ili-Patch16\P_norm",output_path="Ili-Patch16",order=1):
    output_path = os.path.join(output_path,str(order))
    filelist=os.listdir(path)
    random.seed(order)
    random.shuffle(filelist)
    print(filelist)
    trainlist=[]
    vallist=[]
    filenum=len(filelist)
    trainlist=filelist[0:int(0.6*filenum)]
    vallist=filelist[int(0.6*filenum):int(0.8*filenum)]
    testlist=filelist[int(0.8*filenum):]
    train = pd.DataFrame(trainlist)
    train.to_csv(os.path.join(output_path,"train.csv"), sep='\t',header=False, index=False,encoding='utf-8')
    val = pd.DataFrame(vallist)
    val.to_csv(os.path.join(output_path,"val.csv"), sep='\t',header=False, index=False,encoding='utf-8')
    test = pd.DataFrame(testlist)
    test.to_csv(os.path.join(output_path,"test.csv"), sep='\t',header=False, index=False,encoding='utf-8')

order=1
getTrainCsv(path="wuhan-Patch64\P_norm",output_path="wuhan-Patch64/wuhan-split-way",order=order)
# getTrainCsv(path="wuhan-Patch64\P_norm",output_path="wuhan-Patch64")

def aug(path,outpath):
    # 图像归一化
    transform_GY = transforms.ToTensor()#将PIL.Image转化为tensor，即归一化。
    # 图像标准化
    transform_BZ= transforms.Normalize(mean=[0.5,0.5],std=[0.4,0.6])
    #图像尺寸变化
    transform_RS=transforms.Resize([16,16], interpolation=2)
    #中心裁剪
    transform_CC=transforms.CenterCrop(16)
    #随即裁剪
    transform_RC=transforms.RandomCrop(16, padding=0, pad_if_needed=False)
    #随机水平翻转
    transform_RHF=transforms.RandomHorizontalFlip(p=1)
    #随机竖直翻转
    transform_RVF=transforms.RandomVerticalFlip(p=1)
    #随机角度旋转
    transform_RR=transforms.RandomRotation((30,30), resample=False, expand=False)#, center=None
    # degress- 若为单个数，如 30，则表示在（-30， +30）之间随机旋转；若为 sequence，如(30， 60)，则表示在 30-60 度之间随机旋转。
    # resample- 重采样方法选择，可选NEAREST, BILINEAR, BICUBIC，默认为NEAREST
    # expand- True:填满输出图片，False:不填充。
    # center- 可选为中心旋转还是左上角旋转。默认中心旋转

    # transform_compose
    transform_compose_v= transforms.Compose([
    # 先归一化再标准化
        # transform_GY,
        # transform_BZ,
        # transform_RHF,
        transform_RVF,
    ])
    transform_compose_h= transforms.Compose([
    # 先归一化再标准化
        # transform_GY,
        # transform_BZ,
        transform_RHF,
        # transform_RVF,
    ])
    if(os.path.exists(outpath)==False):
        os.makedirs(outpath)
    train_set=os.listdir(path)
    print(train_set)
    for file_name in train_set:
        print(file_name)
        im_data,im_width,im_height,im_bands,im_geotrans,im_proj=readTif(os.path.join(path,file_name))
        writeTiff(im_data,im_width,im_height,im_bands,im_geotrans,im_proj,os.path.join(outpath,file_name))

        img_transform = transform_compose_v(torch.from_numpy(im_data.astype(float)))
        # 输出变换后图像，需要将图像格式调整为PIL.Image形式
        img_after = img_transform .numpy()
        print("img_after.shape",img_after.shape)
        writeTiff(img_after,im_width,im_height,im_bands,im_geotrans,im_proj,os.path.join(outpath,file_name[:-5]+"_v.tiff"))
        
        img_transform = transform_compose_h(torch.from_numpy(im_data.astype(float)))
        # 输出变换后图像，需要将图像格式调整为PIL.Image形式
        img_after = img_transform .numpy()
        print("img_after.shape",img_after.shape)
        writeTiff(img_after,im_width,im_height,im_bands,im_geotrans,im_proj,os.path.join(outpath,file_name[:-5]+"_h.tiff"))
# aug("Ili-Patch16\P_norm","Ili-Patch16\P_norm_aug")
# aug("Ili-Patch16\S_norm","Ili-Patch16\S_norm_aug")
# aug("Ili-Patch16\gt","Ili-Patch16\gt_aug")

# aug("wuhan-Patch64\P_norm","wuhan-Patch64\P_norm_aug")
# aug("wuhan-Patch64\S_norm","wuhan-Patch64\S_norm_aug")
# aug("wuhan-Patch64\gt","wuhan-Patch64\gt_aug")

def csvAddHV(path="Ili-Patch16",order=1):
    path = os.path.join(path,str(order))
    train_set=np.loadtxt(os.path.join(path,"train.csv"),dtype=str)
    train_set.tolist()
    trainlist=[]
    for file_name in train_set:
        # print(file_name)
        trainlist.append(file_name)
        trainlist.append(file_name[:-5]+"_h.tiff")
        trainlist.append(file_name[:-5]+"_v.tiff")
    trainlist = pd.DataFrame(trainlist)
    trainlist.to_csv(os.path.join(path,"trainlist.csv"), sep='\t',header=False, index=False,encoding='utf-8')
csvAddHV("wuhan-Patch64/wuhan-split-way",order)
# csvAddHV("wuhan-Patch64")