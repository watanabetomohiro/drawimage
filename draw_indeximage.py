# パッケージのインストール&インポート gdal
# 取得した画像を表示する。
#
import gc
import math
from osgeo import gdal,gdalconst
from dateutil.parser import parse
from osgeo import gdal_array
from osgeo import osr 
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#from reportlab.pdfgen import canvas
#from reportlab.lib.pagesizes import A4
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio import plot
from rasterio.plot import show
from rasterio.mask import mask
import glob
import geopandas as gpd
import matplotlib.cm as cm

#area_name = 'hitoyoshi'
area_name = 'nagahama'
direct_name = '/home/twatanabe/senti/'
filename = "kohoku_AOI1_clip_paddy" #  長浜
input =  direct_name+area_name+'/INPUT/'
output = direct_name+area_name+'/OUTPUT/'
data_path = input+'/DATA/'
ndpath = output+'IMAGE/NDVI/'
nwpath = output+'IMAGE/NDWI/'
nspath = output+'IMAGE/NDSI/'
gspath = output+'IMAGE/GSI/'
outpath = output+'RESULT/'

gc.collect()

#----------------------------フォルダが存在しない場合に作成する。
def makepath(path):
  if not os.path.exists(path):
    os.makedirs(path)  


# データ入力 (まとめて読込)
data_files = glob.glob(ndpath+'*ndvi.tif')
data_files.sort()
#print(data_files)
len_num = len(data_files)
col = int(4)
#print('len:',len_num)
raw =math.ceil(len_num/col)
my_dpi = 50
im_col =  148 * (col*4)
im_raw = 384 *(col*1)
resize = 1

dwscale_factor = 1/2
print(int(im_raw/my_dpi),int(im_col/my_dpi))

#fig = plt.figure(figsize=(11.69, 8.27))
#fig = plt.figure(figsize=(width, height))
plt.figure(figsize=(int(im_raw/my_dpi),int(im_col/my_dpi)), dpi = my_dpi)
plt.clf()
#plt.figure(figsize=(im_raw, im_col), dpi = my_dpi)
#fig.clf()
#-----------画像の切り出し
for i in range(len_num):
  with rasterio.open(data_files[i]) as src:
    print("num,filename",i+1,data_files[i])  
    # resample data to target shape
    data = src.read(
        out_shape=(
            src.count,
            int(src.height * dwscale_factor),
            int(src.width * dwscale_factor)
        ),
#内挿方法
#        resampling=Resampling.bilinear  # https://rasterio.readthedocs.io/en/latest/topics/resampling.html
        resampling=Resampling.nearest
    )

    # scale image transform
    transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2])
    )
#    print(data.shape)
    
#  j+=1# 画像のプロット位置をシフトさせ配置
  plt.subplot(raw, col, i+1)
  plt.imshow(data[0], clim=(-1, 1), cmap=cm.jet, interpolation='nearest')
#  plt.imshow(img_resize, clim=(-1, 1), cmap=cm.jet, interpolation='nearest')
  plt.colorbar()
  plt.title(data_files[i][-17:-4], fontsize=15)# ファイル名から日付を取得
  plt.tight_layout()

#save
plt.ioff()
strFile = outpath+filename+"_time_series_img1.jpg"
if os.path.isfile(strFile):
    os.remove(strFile)   # Opt.: os.system("rm "+strFile)
plt.savefig(strFile, dpi=my_dpi)

print(src.closed)
#plt.show()
plt.cla() 
plt.close('all')   
plt.close()   
gc.collect()

