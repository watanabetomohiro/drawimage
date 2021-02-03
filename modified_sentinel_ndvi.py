#!/usr/bin/env python
# coding: utf-8
# パッケージのインストール&インポート gdal
#gdalによるNDVI等の計算と出力　https://qiita.com/t-mat/items/24073d8494a7427c0ee1
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
import matplotlib.pyplot as plt
import rasterio
from rasterio import plot
from rasterio.plot import show
from rasterio.mask import mask
import glob
import geopandas as gpd
# matplotlibで時系列図を作成するときには以下をインポートします
from pandas.plotting import register_matplotlib_converters
# これを登録しておきます．
register_matplotlib_converters()
# sklearn(scikit-learn)は機械学習関連のライブラリーです．インポートします．
from sklearn import linear_model
import folium
import matplotlib.cm as cm
# datetimeは日時データを処理する際に便利なメソッドです．インポートします．
from datetime import date, datetime, timedelta
#import datetime
import csv
# 有意検定をするためにscipyのstatsというメソッドをインポートします．
import scipy.stats as stats
#%precision 3
#%matplotlib inline

#area_name = 'hitoyoshi'
area_name = 'nagahama1'
direct_name = '/home/twatanabe/senti/'
filename = "kohoku_AOI1_clip_paddy" #  長浜
input =  direct_name+area_name+'/INPUT/'
output = direct_name+area_name+'/OUTPUT/'
ndpath = output+'IMAGE/NDVI/'
nwpath = output+'IMAGE/NDWI/'
nspath = output+'IMAGE/NDSI/'
gspath = output+'IMAGE/GSI/'
shpath = input+'SHP/'
outpath = output+'RESULT/'
direct = input+'MET/'
# inputデータ
in_file = area_name+"_data1.csv"   # 元の気象データ
input_shp = shpath + filename+'.shp' #　切り出し用のshpファイル
data_path = input+'/DATA/' # オリジナルのセンチネルデータ
# outputデータ
strFile = outpath+filename+"_time_series_img_all.jpg"
merge_file = outpath+filename+"_merge.csv"
corr_file = outpath+filename+"_corr.csv"
desc_file = outpath+filename+"_desc.csv"
precFile = outpath+filename+"_time_series_prec_img.jpg"
precviFile = outpath+filename+"_time_series_prec_vi_img.jpg"
indexFile = outpath+filename+"_time_series_index_img.jpg"
plotFile = outpath+filename+"_time_series_plot_img.jpg"

#----------------------------フォルダが存在しない場合に作成する。
def makepath(path):
  if not os.path.exists(path):
    os.makedirs(path)  
#----------------------------NDVI
# Define a function to calculate NDVI using band arrays for red, NIR bands
def calNDVI(r_band,nir_band):
  nir = nir_band.astype(np.float32)
  red = r_band.astype(np.float32)
  ndvi=np.where(
      (nir+red)==0.,
      0,(nir-red)/(nir+red))
  return ndvi
#----------------------------NDWI 水域
def calNDWI(lir_band,r_band):
  lir = lir_band.astype(np.float32)
  red = r_band.astype(np.float32)
  ndwi=np.where(
      (lir+red)==0., 
      0,(red-lir)/(red+lir))
  return ndwi
#----------------------------NDSI
def calNDSI(lir_band,nir_band):
  lir = lir_band.astype(np.float32)
  nir = nir_band.astype(np.float32)
  ndsi=np.where(
      (lir+nir)==0., 
      0,(lir-nir)/(lir+nir))
  return ndsi
#----------------------------GSI
def calGSI(b_band,g_band,r_band):
  b = b_band.astype(np.float32)
  g = g_band.astype(np.float32)
  r = r_band.astype(np.float32)
  gsi=np.where(
      (b+g+r)==0., 
      0,(r-b)/(b+g+r))
  return gsi
#---------------------------作成データ保存
def outputGeotifSingle(src,epgs,band1,path):
    tfw=src.GetGeoTransform()
    dtype = gdal.GDT_Float32
    band = 1
    width=src.RasterXSize
    height=src.RasterYSize
    output = gdal.GetDriverByName('GTiff').Create(path, width, height, band, dtype)
    output.SetGeoTransform(tfw)
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(epgs)
    output.SetProjection(crs.ExportToWkt())
    output.GetRasterBand(1).WriteArray(band1)
    output.FlushCache()
    output=None
#-----------------------タイトル、平均リスト合計
def dataInfo(data_mean_list, data):
  data_mean = np.nanmean(data)
  data_mean_list = np.append(data_mean_list,data_mean)
  return data_mean_list, data_mean 

#-----------------------アメダスデータの読み込み
# 気象庁アメダスの気温の時系列データを読み込んで，
# DataFrameに割り当てる関数
        # ここがポイント！
        # pandasのread_csvというメソッドでcsvファイルを読み込みます．
        # 引数として，
        # [0]入力ファイル名
        # [1]エンコーディング
        # [2]読み飛ばす行数，
        # [3]column名
        # [4]datetime型で読み込むcolumn名
        # [5]indexとするcolumn名
        # を与える
def readamedas(filename,skipline):
    amedas = pd.read_csv(
        filename, 
        encoding="Shift_JIS", 
        skiprows=skipline, 
#        header=None, 
        names=["Date","Prec","dummy1","dummy2"],
        index_col='Date',
#        parse_dates={'datetime':['date']}, 
        parse_dates=True, 
        )
    return amedas
#def 5.############################################
# ２つの時系列データから時系列図を作成する関数
def timeseries(df,date1,name1,name2,filename):
    # dfのインデックス（時間）をXとする
#    X=df.index
    X=df.loc[:,[date1]].values
    print(df.loc[:,[date1]].values)
    # dfのname1列を指定してデータを取り出し，numpy配列で値をY1に与える．
    Y1=df.loc[:,[name1]].values
    # dfのname1列を指定してデータを取り出し，numpy配列で値をY2に与える．
    Y2=df.loc[:,[name2]].values
    # 時系列図の大きさを指定
    plt.figure(figsize=(20, 10))
    # 1つ目(name1)のグラフを1行1列の1つ目に
    ax1=plt.subplot(1,1,1)
    # 2つ目(name2)のグラフのx軸を共有する
    ax2=ax1.twinx()
    # 1つ目(name1)の時系列 
    ax1.plot(X,Y1,color='blue',label=name1)
    # 2つ目(name2)の時系列 
    ax2.plot(X,Y2,color='red',label=name2)
    # グラフのタイトル
    ax1.set_title("Timeseries:"+name1+" and "+name2)
    # x軸のラベル
    ax1.set_xlabel('Time')
    # y軸（左側の第1軸）のラベル
    ax1.set_ylabel('Index')
    # y軸（右側の第2軸）のラベル
    ax2.set_ylabel('Amount of Precipitation [mm/hr]')
    # 1つ目(name1)の凡例（左上に置く） 
    ax1.legend(loc='upper left')
    # 2つ目(name1)の凡例（右上に置く）
    ax2.legend(loc='upper right')
    # 保存するファイル名
    plt.savefig(filename)
    # 図を閉じる
    plt.close()
    return
#-----------------------散布図
# ２つの時系列データから散布図を作成する関数
def scatter(df,name1,name2,filename):
    # ここがポイント！
    # scikit-learnの線形回帰モデルのクラスを呼び出す
    clf = linear_model.LinearRegression()
    # 説明変数Xにはname1を割り当てる（numpy配列）
    X=df.loc[:,[name1]].values
    # 説明変数Yにはname2を割り当てる（numpy配列）
    Y=df.loc[:,[name2]].values
    # ここがポイント！
    # Y=aX+bの予測モデルを作成する
    clf.fit(X,Y)
    # ここがポイント！
    # 回帰係数a
    slope=clf.coef_
    # ここがポイント！
    # 切片b
    intercept=clf.intercept_
    # ここがポイント！
    # 決定係数R2（回帰直線の当てはまりの良さ）
    r2=clf.score(X,Y)
    # 文字列"Y=aX+b (R2=r2)"
    equation="  y = "+str('{:.1f}'.format(slope[0][0]))+" x +"+str('{:.0f}'.format(intercept[0]))+" (R2="+str('{:.2f}'.format(r2))+")"
    print(equation)
    # 相関係数とその有意確率p-値を計算
    corrcoef, pvalue = stats.pearsonr(np.ravel(X),np.ravel(Y))
    # 散布図の大きさを指定
    plt.figure(figsize=(8, 8))
    # 散布図のプロット
    plt.plot(X, Y, 'o')
    # ここがポイント！
    # 散布図上に回帰直線を引く
    plt.plot(X, clf.predict(X))
    # 文字列"Y=aX+b (R2=r2)"を図の左上に置く
    plt.text(np.nanmin(X), np.nanmax(Y), equation)
    # グラフのタイトル
    plt.title("Scatter diagram:"+name1+" and "+name2)
    # x軸のラベル
    plt.xlabel(name1)
#    plt.xlim(-1, 1) #y軸の最小と最大を決める
    # y軸のラベル
    plt.ylabel(name2)
#    if os.path.isfile(plotFile):
#        os.remove(plotFile)   # Opt.: os.system("rm "+strFile)
    plt.savefig(filename)
#    plt.ylim(-1, 1) #y軸の最小と最大を決める
    # 図を閉じる
    plt.close()
    return corrcoef, pvalue

# ディレクトリが存在しない場合、ディレクトリを作成する 
makepath(ndpath)
makepath(nwpath)
makepath(nspath)
makepath(gspath)
makepath(outpath)
makepath(output)

# データ入力 (まとめて読込)
data_files = glob.glob(data_path+'*V.tif')
data_files.sort()

# シェープファイル読込（encoding='SHIFT-JIS'とすることで日本語データにも対応）
df_shp = gpd.read_file(input_shp,encoding='SHIFT-JIS')
print("input_shp:",input_shp)

#----------------------グラフ用リストの初期化
ndvi_mean_list = np.zeros(0)
ndwi_mean_list = np.zeros(0)
ndsi_mean_list = np.zeros(0)
gsi_mean_list = np.zeros(0)
data_title_list = np.zeros(0)

# ----------------------画像範囲設定
len_num = len(data_files)
col = int(4)
raw =math.ceil(len_num)
print("raw:",raw)
my_dpi = 50
im_col =  148 * (col*2*9)
im_raw = 384 *(col*1)


#fig = plt.figure(figsize=(18, image_len))
#fig = plt.figure(figsize=(25, 200))
plt.figure(figsize=(int(im_raw/my_dpi),int(im_col/my_dpi)), dpi = my_dpi)
plt.clf()
print(mpl.matplotlib_fname())

j=0
num=1
#-----------画像の切り出し
#for i in range(10):
for i in range(len_num):
  input_raster = data_files[i]
  output_raster = data_files[i][:-4]+'_clip.tif'
  print(input_raster)
  print(output_raster)
 # gdal.Warp(output_raster, input_raster, format = 'GTiff', cutlineDSName = input_shp, dstNodata = np.nan)


#----read red datra--------
  src = gdal.Open(output_raster, gdal.GA_ReadOnly)
  src.RasterXSize # 水平方向ピクセル数
  src.RasterYSize # 鉛直方向ピクセル数
  src.RasterCount # バンド数

  #　青、緑、赤、NIR、SWIR（よく使うバンド）はそれぞれ B2, B3, B4, B8, B12（またはB11）に相当
  #  band_name = ['B2', 'B3', 'B4', 'B8', 'B12']
  barr = src.GetRasterBand(1).ReadAsArray() # 第1バンド blue
  garr = src.GetRasterBand(2).ReadAsArray() # 第2バンド green
  rarr = src.GetRasterBand(3).ReadAsArray() # 第3バンド red
  narr = src.GetRasterBand(4).ReadAsArray() # 第8バンド 近赤外
  swarr = src.GetRasterBand(5).ReadAsArray() # 第12バンド 短波長

#----get EPGS iinfo---------
  proj = int(osr.SpatialReference(wkt=src.GetProjection()).GetAttrValue('AUTHORITY',1))
#  print('proj_type=',type(proj),'proj=',proj)

#-- Call the ndvi(),swvi() function on red, NIR bands
  ndvi = calNDVI(rarr, narr)
  ndwi = calNDWI(swarr, rarr)
  ndsi = calNDSI(swarr, narr)
  gsi = calGSI(barr, garr, rarr)

#---calc ndvi ndwi mean----- 
  ndvi_mean_list, ndvi_mean = dataInfo(ndvi_mean_list, ndvi)
  ndwi_mean_list, ndwi_mean = dataInfo(ndwi_mean_list, ndwi)
  ndsi_mean_list, ndsi_mean = dataInfo(ndsi_mean_list, ndsi)
  gsi_mean_list, gsi_mean = dataInfo(gsi_mean_list, gsi)

#---get image date------
#  data_title = data_files[i][-31:-23]
  data_title = input_raster[-26:-18]
  data_title_list = np.append(data_title_list, data_title)
  print(num,data_title,'ndvi_mean=',ndvi_mean,'ndwi_mean=',ndwi_mean,'ndsi_mean=',ndsi_mean,'gsi_mean=',gsi_mean)

#-- Output NDVI data---- 
  outputGeotifSingle(src,proj,ndvi,ndpath+filename+'_'+data_title+'_ndvi.tif')
  outputGeotifSingle(src,proj,ndwi,nwpath+filename+'_'+data_title+'_ndwi.tif')
  outputGeotifSingle(src,proj,ndsi,nspath+filename+'_'+data_title+'_ndsi.tif')
  outputGeotifSingle(src,proj,gsi,gspath+filename+'_'+data_title+'_gsi.tif')

#---Draw Images ---
  num+=1
  print("num:",num)
 # 色の情報(https://matplotlib.org/tutorials/colors/colormaps.html)
  j+=1
  plt.subplot(raw,col,j);plt.imshow( ndvi, clim=(-1, 1), cmap=cm.jet, interpolation='nearest')
  plt.colorbar()
  plt.xlabel(data_title+'_ndvi', fontsize=30)        #x軸に名前をつける
  j+=1
  plt.subplot(raw,col,j);plt.imshow( ndwi, clim=(-1, 1), cmap=cm.jet, interpolation='nearest')
  plt.colorbar()
  plt.xlabel(data_title+'_ndwi', fontsize=30)
  j+=1
  plt.subplot(raw,col,j);plt.imshow( ndsi, clim=(-1, 1), cmap=cm.jet, interpolation='nearest')
  plt.colorbar()  
  plt.xlabel(data_title+'_ndsi', fontsize=30)
  j+=1  
  plt.subplot(raw,col,j);plt.imshow( gsi, clim=(-1, 1), cmap=cm.jet, interpolation='nearest')
  plt.colorbar()
  plt.xlabel(data_title+'_gsi', fontsize=30)
  plt.tight_layout()

##----画像の保存
#plt.show()
plt.ioff()
if os.path.isfile(strFile):
        os.remove(strFile)   # Opt.: os.system("rm "+strFile)
plt.savefig(strFile, dpi=my_dpi)
plt.cla()
#plt.tight_layout()
#plt.show()

#----------------------------------------------------
#気象データの読み込み　　ここからグラフとcsvの作成と相関係数を求める。
# 降水量の合計日数（移動合計）
interval3 =3
interval4 =4
interval5 =5
interval6 =6
interval7 =7
interval30 =30

#気象データ読み込み＋ヘッダ変更
skipline1=5
df = readamedas(direct+in_file,skipline1)
# DataFrame(amedas)の中のdummy1とdummy2の列を削除する．
df=df.drop(['dummy1','dummy2'],axis=1)
print("df",df)

# パラメータ設定
date_time = "Date"
prec = "Prec"
prec3 = "Prec_3days"
prec4 = "Prec_4days"
prec5 = "Prec_5days"
prec6 = "Prec_6days"
prec7 = "Prec_7days"
prec30 = "Prec_30days"
name_NDVI = "NDVI"
name_NDWI = "NDWI"
name_NDSI = "NDSI"
name_GSI = "GSI"

# 年月日を日付に変換、フォーマットもint型にする。
#df['年月日'] = pd.to_datetime(df['年月日'])
#df['年月日'] =  pd.Series(df['年月日'].dt.strftime('%Y%m%d'), dtype='str')
df[prec] =  pd.Series(df[prec], dtype='float') #floatに変換 
# 日降水量の移動合計を追加　（移動平均のような）
df_sum3 = df[prec].rolling(window=interval3).sum()
df_sum4 = df[prec].rolling(window=interval4).sum()
df_sum5 = df[prec].rolling(window=interval5).sum()
df_sum6 = df[prec].rolling(window=interval6).sum()
df_sum7 = df[prec].rolling(window=interval7).sum()
df_sum30 = df[prec].rolling(window=interval30).sum()
## 移動合計データをdf リストに結合
df[prec3] = df_sum3
df[prec4] = df_sum4
df[prec5] = df_sum5
df[prec6] = df_sum6
df[prec7] = df_sum7
df[prec30] = df_sum30

#--- NDVI等の計算結果をpandasでリスト化
ct_csv = np.array([data_title_list, ndvi_mean_list, ndwi_mean_list, ndsi_mean_list, gsi_mean_list]).T #行列を転置
vi_data = pd.DataFrame(ct_csv,columns =[date_time, name_NDVI ,name_NDWI, name_NDSI,name_GSI]) #タイトル行を追加 
#print("vidata",vi_data)
# 年月日を日付に変換、フォーマットもint型にする。
vi_data[date_time] = pd.to_datetime(vi_data[date_time])
#vi_data = vi_data.set_index([date_time]) # index keyの設定
# ndvi等のフォーマットもfloat型にする。
vi_data[name_NDVI] =  pd.Series(vi_data[name_NDVI], dtype='float') #floatに変換 
vi_data[name_NDWI] =  pd.Series(vi_data[name_NDWI], dtype='float') #floatに変換
vi_data[name_NDSI] =  pd.Series(vi_data[name_NDSI], dtype='float') #floatに変換
vi_data[name_GSI] =  pd.Series(vi_data[name_GSI], dtype='float') #floatに変換

#　同じ日付を抽出　https://reffect.co.jp/python/python-pandas-not-duplicate-in-two-excels
df_merge = pd.merge(df,vi_data,on=date_time,how="inner",indicator=True) # how=outerはNANが残る。
#df_merge = pd.merge(df, vi_data, how="inner",left_index=True, right_index=True) # how=outerはNANが残る。
'''
left_index=Trueと設定すると，左側のデータフレームのインデックスを結合のキーとして用います．right_index=Trueと設定すると，右側のデータフレームのインデックスを結合のキーとして用います．
'''
df_clip = df_merge.dropna(how='any') #NaN（欠損値）が一つでもある行は削除する。（https://note.nkmk.me/python-pandas-nan-dropna-fillna/）
df_clip[date_time] =  pd.Series(df_clip[date_time].dt.strftime('%Y%m%d'), dtype='int') # 日時のフォーマットを変更
#df_clip[date_time] =  pd.Series(df_clip[date_time].dt.strftime('%Y%m%d'), dtype='str') # 日時のフォーマットを変更
#df_clip = df_clip.set_index([date_time]) # index keyの設定
print(df_clip)

# 処理されたデータを用いて散布図を作成し、図として保存する．
#corrcoef, pvalue = scatter(df_clip,name_NDWI,name_NDVI,plotFile)
corrcoef, pvalue = scatter(df_clip,name_NDVI,prec5,plotFile)
print("Peoc_NDVI... Corr=",corrcoef,"(p-value=",pvalue,")")
# 処理されたデータを用いて時系列図を作成します．
# ファイル名：precviFile # 
timeseries(df_clip,date_time,name_NDVI,prec5,precviFile)


#'統計量を算出してcsvで保存する。
df_clip_desc = df_clip.describe()
#print(df_clip.describe())
if os.path.isfile(desc_file):
        os.remove(desc_file)   # Opt.: os.system("rm "+strFile)
df_clip_desc.to_csv(desc_file, encoding="Shift_JIS",  date_format='%Y%m%d')

# mergeしたデータをcsvで保存する。
if os.path.isfile(merge_file):
        os.remove(merge_file)   # Opt.: os.system("rm "+strFile)
df_clip.to_csv(merge_file, encoding="Shift_JIS", date_format='%Y%m%d')


## 相関係数をまとめて計算してcsvで保存する。
#'pearson': ピアソンの積率相関係数（デフォルト）
#'kendall': ケンドールの順位相関係数
#'spearman': スピアマンの順位相関係数
# 表の並べ替え　https://qiita.com/Masahiro_T/items/2f9574c80193f58af7fe
df_clip_corr = df_clip.corr(method='pearson')
#print(df_clip.corr())
if os.path.isfile(corr_file):
        os.remove(corr_file)   # Opt.: os.system("rm "+strFile)
df_clip_corr.to_csv(corr_file, encoding="Shift_JIS",  date_format='%Y%m%d')


date1 =  np.array(df_clip[date_time])
ndvi1 = np.array(df_clip[name_NDVI])
ndwi1 = np.array(df_clip[name_NDWI])
ndsi1 = np.array(df_clip[name_NDSI])
gsi1 = np.array(df_clip[name_GSI])
np_prec = np.array(df_clip[prec])
np_prec3 = np.array(df_clip[prec3])
np_prec4 = np.array(df_clip[prec4])
np_prec5 = np.array(df_clip[prec5])
np_prec6 = np.array(df_clip[prec6])
np_prec7 = np.array(df_clip[prec7])
np_prec30 = np.array(df_clip[prec30])

##グラフを書く
dfs = pd.DataFrame(df_clip[prec])
sum = dfs.size
print("リスト数",sum)


#降水量をグラフに示す。
fig1,ax1 = plt.subplots(figsize=(15,3))
plt.plot(np_prec, marker='o', label='PREC')
plt.plot(np_prec3, marker='*', label='PREC3')
plt.plot(np_prec4, marker='+', label='PREC4')
plt.plot(np_prec5, marker='.', label='PREC5')
plt.plot(np_prec6, marker='1', label='PREC6')
plt.plot(np_prec7, marker='2', label='PREC7')
plt.plot(np_prec30, marker='3', label='PREC30')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=10)
#plt.legend()
ax1.set_xticks(np.arange(0,sum))  #X軸の数
ax1.set_xticklabels(date1, fontsize=10, rotation = 25, ha="center")
plt.tight_layout()
plt.grid(True)
#plt.show()
if os.path.isfile(precFile):
        os.remove(precFile)   # Opt.: os.system("rm "+strFile)
plt.savefig(precFile)
plt.cla()

#NDVIの時系列変化をグラフに示す
fig2,ax2 = plt.subplots(figsize=(15,3))
#plt.plot(prep, marker='o', label='PREP')
plt.plot(ndvi1, marker='*', label=name_NDVI)
plt.plot(ndwi1, marker='+', label=name_NDWI)
plt.plot(ndsi1, marker='.', label=name_NDSI)
plt.plot(gsi1, marker='1', label=name_GSI)
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=10)
#plt.legend()
ax2.set_xticks(np.arange(0,sum))  #X軸の数
ax2.set_xticklabels(date1, fontsize=10,rotation = 25, ha="center")
ax2.set_ylim(-1, 1) #y軸の最小と最大を決める
plt.tight_layout() #レイアウトを最適化　ラベルが消えるのを制御する。
plt.grid(True)
#plt.show()
if os.path.isfile(indexFile):
        os.remove(indexFile)   # Opt.: os.system("rm "+strFile)
plt.savefig(indexFile)
plt.cla()
