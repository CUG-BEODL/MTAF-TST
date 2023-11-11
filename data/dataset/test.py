import os
import numpy as np
from osgeo import gdal
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"ÎÄ¼þÎÞ·¨´ò¿ª")
        return
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    return im_data,im_width,im_height,im_bands,im_geotrans,im_proj
#±£´ætifÎÄ¼þº¯Êý
from osgeo import gdal
import numpy as np
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
        #´´½¨ÎÄ¼þ
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #Ð´Èë·ÂÉä±ä»»²ÎÊý
        dataset.SetProjection(im_proj) #Ð´ÈëÍ¶Ó°
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

path="data/dataset/P_S_16_proj_noverlap_norm/S"
outpath="data/dataset/P_S_16_proj_noverlap_norm_5-10/S"
for filename in os.listdir(path):
    print(filename)
    im_data,im_width,im_height,im_bands,im_geotrans,im_proj=readTif(os.path.join(path,filename))
    im_data_new=im_data[16:-60,:,:]
    print(im_data_new.shape)
    writeTiff(im_data_new,im_width,im_height,im_bands,im_geotrans,im_proj,os.path.join(outpath,filename))









