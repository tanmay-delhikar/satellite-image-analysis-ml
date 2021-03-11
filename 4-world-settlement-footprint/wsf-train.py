import argparse
import codecs
import json
import multiprocessing
import os
import pickle
import subprocess

from sentinelsat.sentinel import SentinelAPI
from tqdm import tqdm

import gdal
import numpy as np
import rasterio
from gdalconst import GA_ReadOnly
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.externals.joblib import parallel_backend



def download_scene(file_id, out_path):
    api = SentinelAPI('<uname>', '<pwd>',
                      api_url='https://scihub.copernicus.eu/apihub')
    return api.download(file_id, directory_path=out_path)


def unzip_scene(file_name, out_dir):
    subprocess.call('unzip '+file_name+' -d '+out_dir, shell=True)

def check_or_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def jp2_to_tif(input_file, output_file):
    subprocess.call('gdal_translate -tr 30 30 ' + input_file+' '+output_file, shell=True)
    
def band_sorter_S2(list_temp):
  sorted_list=[]
  for element in list_temp:
    if element.endswith('_B01.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B02.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B03.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B04.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B05.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B06.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B07.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B08.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B09.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B10.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B11.tif'):
      sorted_list.append(element)
  for element in list_temp:
    if element.endswith('_B12.tif'):
      sorted_list.append(element) 
  for element in list_temp:
    if element.endswith('_B8A.tif'):
      sorted_list.append(element) 
  return sorted_list


def stack_bands(file_list, output_file):
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    meta.update(count=len(file_list))

    with rasterio.open(output_file, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


def clip_from_vrt(template_tif, vrt_file, temp_out_path, output_file):
    datag = gdal.Open(template_tif, GA_ReadOnly)
    geoTransform = datag.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * datag.RasterXSize
    miny = maxy + geoTransform[5] * datag.RasterYSize
    epsg = (gdal.Info(template_tif, format='json')[
            'coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
    ds = gdal.Translate(temp_out_path, vrt_file, projWin=[
                        minx, maxy, maxx, miny], projWinSRS='EPSG:'+epsg)
    data = ds.ReadAsArray()
    create_new_tif(template_tif, output_file, data, 'uint8', 1)
    ds = None
    datag = None

def create_new_tif(sourceraster, targetraster, array2d, dtype, nbands):
    with rasterio.open(sourceraster) as src:
        metadata = src.profile
    metadata['count'] = nbands
    metadata['dtype'] = dtype
    with rasterio.open(targetraster, 'w', **metadata) as dst:
        dst.write(array2d, 1)
        print('New tif created at: ', str(targetraster))

#CALCULATE SPECTRAL INDICES LIKE NDVI, NDBI etc..
def calculate_spectral_index(band1,band2):
    ndbi=np.where((band1+band2)==0.,0,(band1-band2)/(band1+band2))
    return ndbi

def read_band_data(fileName):
    a=gdal.Open(fileName)
    a1=a.GetRasterBand(1).ReadAsArray().astype(dtype='float')
    return a1


#CALCULATE SPECTRAL INDICES FOR A SINGLE SCENE
def calculate_all_indices_for_a_scene(filename):
    a=gdal.Open(filename)
    temp_list=[]
    b2=a.GetRasterBand(2).ReadAsArray().astype(dtype='float')
    b3=a.GetRasterBand(3).ReadAsArray().astype(dtype='float')
    b4=a.GetRasterBand(4).ReadAsArray().astype(dtype='float')
    b5=a.GetRasterBand(5).ReadAsArray().astype(dtype='float')
    b6=a.GetRasterBand(6).ReadAsArray().astype(dtype='float')
    b7=a.GetRasterBand(7).ReadAsArray().astype(dtype='float')

    temp_list.append(calculate_spectral_index(b6,b5))
    temp_list.append(calculate_spectral_index(b3,b5))
    temp_list.append(calculate_spectral_index(b5,b4))
    temp_list.append(calculate_spectral_index(b6,b7))
    temp_list.append(calculate_spectral_index(b4,b2))
    temp_list.append(calculate_spectral_index(b3,b2))
    return temp_list

#CALCULATE TEMPORAL STATISTICS LIKE MEAN, MIN, MAX etc..
def calc_temporal_statistics(index,scene1,scene2,scene3,size,return_dict):
    print('Process started: '+str(index))
    temp_min = np.zeros(shape=(size,size))
    temp_max = np.zeros(shape=(size,size))
    temp_mean = np.zeros(shape=(size,size))
    temp_std = np.zeros(shape=(size,size))
    temp_mean_slope = np.zeros(shape=(size,size))


    for i,j in tqdm(np.ndindex(temp_min.shape)):    
        x=scene1[i][j]
        y=scene2[i][j]
        z=scene3[i][j]

        temp_min[i][j]=np.amin([x,y,z])
        temp_max[i][j]=np.amax([x,y,z])
        temp_mean[i][j]=np.mean([x,y,z])
        temp_std[i][j]=np.std([x,y,z])  
        temp_mean_slope[i][j]=np.average(np.diff([x,y,z]))
        # if i%10000==0 and j%10000==0:
        #     print("Iteration done: "+str(i)+' for index: '+str(index))

    return_dict[index]=(temp_min,temp_max,temp_mean,temp_std,temp_mean_slope)


def main():
    parser = argparse.ArgumentParser(description="WSF training")
    parser.add_argument("-of", "--outputPath", help="Output path to save preprocessed files")
    parser.add_argument("-c", "--config", help="Full file path of config.json")
    parser.add_argument("-if", "--inputPath", help="Input path to save preprocessed files")


    args = parser.parse_args()
    output_path = args.outputPath
    input_path=args.inputPath
    config_file_path = args.config

    if(output_path is None or input_path is None):
        print("Output/Input path is missing")
        exit()

    if(config_file_path is None):
        print("Config path is missing")
        exit()

    check_or_create_directory(output_path)

    #DOWNLOAD TRAINING DATA

    #scene_ids=['03cf3314-1b05-46f1-af1b-cee2652edf2b','4f07815d-5709-4b0b-8fdd-7e12c78e6546','5edb9725-8699-4cea-b0e3-7e582d9e68d6']
    safe_folder_paths_base_names=next(os.walk(input_path))[1]
    safe_folder_paths_base_names.sort()
    print(safe_folder_paths_base_names)
    for i in safe_folder_paths_base_names:
        if '.SAFE' not in i:
            safe_folder_paths_base_names.remove(i)

    # for i in scene_ids:
    #     metadata = download_scene(i, output_path)
    #     unzip_scene(metadata['path'], output_path)
    #     safe_folder_paths.append(os.path.join(output_path, metadata['title']+'.SAFE'))
    

    data_directory=os.path.join(output_path,'data')
    check_or_create_directory(data_directory)

    temporal_files=[]

    for index,safe_folder in enumerate(safe_folder_paths_base_names):
            temp_out_directory=os.path.join(output_path,safe_folder)
            check_or_create_directory(temp_out_directory)
            for root, dir, filelist in os.walk(os.path.join(input_path,safe_folder)):
                if root.endswith('IMG_DATA'):
                    img_data_path = root
            print('IMG DATA: ',img_data_path)
            for i in os.listdir(img_data_path):
                if(i.endswith('.jp2')):
                    print('Processing...'+i)
                    jp2_to_tif(os.path.join(img_data_path, i), os.path.join(temp_out_directory, os.path.splitext(i)[0]+'.tif'))

            tif_files_temp = []
            for i in os.listdir(temp_out_directory):
                if i.endswith('.tif') and '_B' in i:
                    tif_files_temp.append(os.path.join(temp_out_directory, i))

            tif_files_temp.sort()
            tif_files=band_sorter_S2(tif_files_temp)
            print('Sorted list: ',tif_files)
            stack_bands(tif_files, os.path.join(output_path,'t'+str(index)+'.tif'))
            temporal_files.append(os.path.join(output_path,'t'+str(index)+'.tif'))
            print('Index done',index)

    #CLIP REFERENCE DATA
    configJson = json.load(codecs.open(config_file_path, 'r', 'utf-8-sig'))
    vrt_file=configJson['commons']['original-built']
    clip_from_vrt(temporal_files[0], vrt_file, os.path.join(data_directory,'trash.tif'), os.path.join(data_directory,'Y.tif'))

    with rasterio.open(temporal_files[0]) as src1:
        prof = src1.profile
        if(prof['width']!=prof['height']):
            print('Input data does not have equal width and height, they must be same')
            exit()
        IMG_SIZE = prof['width']

    #SCENE INDICES
    scene_indices=[]
    for index, filename in enumerate(temporal_files):
        scene_indices.append(calculate_all_indices_for_a_scene(filename))

    #CALULATE TEMPORAL INDICES LIKE MEAN, SLOPE ETC.
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    pool = multiprocessing.Pool(5)
    for i in range(0,6):
        pool.apply_async(calc_temporal_statistics, args=(i,scene_indices[0][i],scene_indices[1][i],scene_indices[2][i],IMG_SIZE,return_dict))

    pool.close()
    pool.join()
    print(return_dict.keys())

    ndbi_statistics=return_dict.get(0)
    mndwi_statistics=return_dict.get(1)
    ndvi_statistics=return_dict.get(2)
    ndmir_statistics=return_dict.get(3)
    ndrb_statistics=return_dict.get(4)
    ndgb_statistics=return_dict.get(5)

    tif2=gdal.Open(os.path.join(data_directory,'Y.tif'))
    Y_raster = tif2.GetRasterBand(1).ReadAsArray()
    print(Y_raster.shape)
    Y_raster[Y_raster==0]=125
    Y_raster[Y_raster==1]=255
    Y_raster[Y_raster==2]=125
    Y_raster[Y_raster==3]=0
    Y_raster[Y_raster==4]=0
    Y_raster[Y_raster==5]=0
    Y_raster[Y_raster==6]=0
    Y_raster[Y_raster==125]=1
    Y_raster[Y_raster==255]=1
    print("UNIQUE CLASSES: ", np.unique(Y_raster))
    Y_list=Y_raster.flatten().tolist()
    Y=np.reshape(Y_list,(IMG_SIZE*IMG_SIZE,1))
    print(Y.shape)


    records=[]
    for i in tqdm(range(0,5)):
            records.append(ndbi_statistics[i].flatten().tolist())
            records.append(mndwi_statistics[i].flatten().tolist())
            records.append(ndvi_statistics[i].flatten().tolist())
            records.append(ndmir_statistics[i].flatten().tolist())
            records.append(ndrb_statistics[i].flatten().tolist())
            records.append(ndgb_statistics[i].flatten().tolist())

    record=np.column_stack(records)
    X=np.append(record, Y, axis=1)
    print(X.shape)
    return_dict=None
    scene_indices=None
    x0=X[np.where(X[:,30] == 0)]
    x1=X[np.where(X[:,30] == 1)]
    X0=x0[np.random.choice(np.shape(x0)[0],5000, replace=False), :]
    X1=x1[np.random.choice(np.shape(x1)[0],5000, replace=False), :]
    X_small=np.append(X0,X1,axis=0)

    X_train=X_small[:,0:30]
    Y_train=X_small[:,30]

    parameter_candidates = [{'C': [0.1,1,10,100], 'gamma': [0.1,1,10,50,100], 'kernel': ['rbf']}]
    classifier = GridSearchCV(SVC(),cv=5, param_grid=parameter_candidates,verbose=True,n_jobs=20)
    classifier.fit(X_train, Y_train)  
    print('Best score:', classifier.best_score_) 
    print('Best C value:',classifier.best_estimator_.C) 
    print('Best Gamma value:',classifier.best_estimator_.gamma)

    svm = SVC(kernel='rbf',C=classifier.best_estimator_.C,gamma=classifier.best_estimator_.gamma,probability=True,verbose=True)
    svm.fit(X_train, Y_train)
    with open(os.path.join(data_directory,'model.pkl'), 'wb') as f:
        pickle.dump(svm, f)
    print('Training done!, Model dumped at: ',os.path.join(data_directory,'model.pkl'))



if __name__ == '__main__':
    main()
