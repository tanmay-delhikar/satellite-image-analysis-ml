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


def create_new_tif(sourceraster, targetraster, array2d, dtype, nbands):
    with rasterio.open(sourceraster) as src:
        metadata = src.profile
    metadata['count'] = nbands
    metadata['dtype'] = dtype
    with rasterio.open(targetraster, 'w', **metadata) as dst:
        dst.write(array2d, 1)
        print('New tif created at: ', str(targetraster))


def check_or_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def jp2_to_tif(input_file, output_file):
    subprocess.call('gdal_translate -tr 30 30 ' +
                    input_file+' '+output_file, shell=True)


def band_sorter_S2(list_temp):
    sorted_list = []
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


# CALCULATE SPECTRAL INDICES LIKE NDVI, NDBI etc..
def calculate_spectral_index(band1, band2):
    ndbi = np.where((band1+band2) == 0., 0, (band1-band2)/(band1+band2))
    return ndbi


def read_band_data(fileName):
    a = gdal.Open(fileName)
    a1 = a.GetRasterBand(1).ReadAsArray().astype(dtype='float')
    return a1


# CALCULATE SPECTRAL INDICES FOR A SINGLE SCENE
def calculate_all_indices_for_a_scene(filename):
    a = gdal.Open(filename)
    temp_list = []
    b2 = a.GetRasterBand(2).ReadAsArray().astype(dtype='float')
    b3 = a.GetRasterBand(3).ReadAsArray().astype(dtype='float')
    b4 = a.GetRasterBand(4).ReadAsArray().astype(dtype='float')
    b5 = a.GetRasterBand(5).ReadAsArray().astype(dtype='float')
    b6 = a.GetRasterBand(6).ReadAsArray().astype(dtype='float')
    b7 = a.GetRasterBand(7).ReadAsArray().astype(dtype='float')

    temp_list.append(calculate_spectral_index(b6, b5))
    temp_list.append(calculate_spectral_index(b3, b5))
    temp_list.append(calculate_spectral_index(b5, b4))
    temp_list.append(calculate_spectral_index(b6, b7))
    temp_list.append(calculate_spectral_index(b4, b2))
    temp_list.append(calculate_spectral_index(b3, b2))
    return temp_list

# CALCULATE TEMPORAL STATISTICS LIKE MEAN, MIN, MAX etc..


def calc_temporal_statistics(index, scene1, scene2, scene3, size, return_dict):
    print('Process started: '+str(index))
    temp_min = np.zeros(shape=(size, size))
    temp_max = np.zeros(shape=(size, size))
    temp_mean = np.zeros(shape=(size, size))
    temp_std = np.zeros(shape=(size, size))
    temp_mean_slope = np.zeros(shape=(size, size))

    for i, j in tqdm(np.ndindex(temp_min.shape)):
        x = scene1[i][j]
        y = scene2[i][j]
        z = scene3[i][j]

        temp_min[i][j] = np.amin([x, y, z])
        temp_max[i][j] = np.amax([x, y, z])
        temp_mean[i][j] = np.mean([x, y, z])
        temp_std[i][j] = np.std([x, y, z])
        temp_mean_slope[i][j] = np.average(np.diff([x, y, z]))
        # if i%10000==0 and j%10000==0:
        #     print("Iteration done: "+str(i)+' for index: '+str(index))

    return_dict[index] = (temp_min, temp_max, temp_mean,
                          temp_std, temp_mean_slope)



def main():
    parser = argparse.ArgumentParser(description="WSF based results")
    parser.add_argument("-of", "--outputFolder",
                        help="Output path to save preprocessed files")
    parser.add_argument("-if", "--inputFolder",
                        help="Input path to save preprocessed files")
    parser.add_argument("-c", "--config", help="Full file path of config.json")

    args = parser.parse_args()
    input_folder = args.inputFolder
    output_folder = args.outputFolder
    config_file_path = args.config

    if(output_folder is None or input_folder is None):
        print("Output/Input path is missing")
        exit()

    if(config_file_path is None):
        print("Config path is missing")
        exit()

    check_or_create_directory(output_folder)

    safe_folder_paths_base_names = next(os.walk(input_folder))[1]
    safe_folder_paths_base_names.sort()
    print(safe_folder_paths_base_names)
    for i in safe_folder_paths_base_names:
        if '.SAFE' not in i:
            safe_folder_paths_base_names.remove(i)

    data_directory = os.path.join(output_folder, 'WSF')
    check_or_create_directory(data_directory)

    temporal_files = []

    for index, safe_folder in enumerate(safe_folder_paths_base_names):
        temp_out_directory = os.path.join(output_folder, safe_folder)
        check_or_create_directory(temp_out_directory)
        for root, dir, filelist in os.walk(os.path.join(input_folder, safe_folder)):
            if root.endswith('IMG_DATA'):
                img_data_path = root
        print('IMG DATA: ', img_data_path)
        for i in os.listdir(img_data_path):
            if(i.endswith('.jp2')):
                print('Processing...'+i)
                jp2_to_tif(os.path.join(img_data_path, i), os.path.join(
                    temp_out_directory, os.path.splitext(i)[0]+'.tif'))

        tif_files_temp = []
        for i in os.listdir(temp_out_directory):
            if i.endswith('.tif') and '_B' in i:
                tif_files_temp.append(os.path.join(temp_out_directory, i))

        tif_files_temp.sort()
        tif_files = band_sorter_S2(tif_files_temp)
        print('Sorted list: ', tif_files)
        stack_bands(tif_files, os.path.join(
            output_folder, 't'+str(index)+'.tif'))
        temporal_files.append(os.path.join(
            output_folder, 't'+str(index)+'.tif'))
        print('Index done', index)

    configJson = json.load(codecs.open(config_file_path, 'r', 'utf-8-sig'))
    model = configJson['wsf-built']['test']['model-path']

    with rasterio.open(temporal_files[0]) as src1:
        prof = src1.profile
        if(prof['width'] != prof['height']):
            print('Input data does not have equal width and height, they must be same')
            exit()
        IMG_SIZE = prof['width']
    # SCENE INDICES
    scene_indices = []
    for index, filename in enumerate(temporal_files):
        scene_indices.append(calculate_all_indices_for_a_scene(filename))

    # CALULATE TEMPORAL INDICES LIKE MEAN, SLOPE ETC.
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(0, 6):
        p = multiprocessing.Process(target=calc_temporal_statistics, args=(
            i, scene_indices[0][i], scene_indices[1][i], scene_indices[2][i], IMG_SIZE, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
        proc.terminate()
        print(return_dict.keys())

    ndbi_statistics = return_dict.get(0)
    mndwi_statistics = return_dict.get(1)
    ndvi_statistics = return_dict.get(2)
    ndmir_statistics = return_dict.get(3)
    ndrb_statistics = return_dict.get(4)
    ndgb_statistics = return_dict.get(5)

    records = []
    for i in tqdm(range(0, 5)):
        records.append(ndbi_statistics[i].flatten().tolist())
        records.append(mndwi_statistics[i].flatten().tolist())
        records.append(ndvi_statistics[i].flatten().tolist())
        records.append(ndmir_statistics[i].flatten().tolist())
        records.append(ndrb_statistics[i].flatten().tolist())
        records.append(ndgb_statistics[i].flatten().tolist())

    X = np.column_stack(records)
    print('FULL TESTING DATA SHAPE: ')
    print(X.shape)

    svclassifier = pickle.load(open(model, 'rb'))
    Y_pred = svclassifier.predict(X)
    results = svclassifier.predict_proba(X)

    class_0_probs = []
    for i in results:
        class_0_probs.append(i[0])

    create_new_tif(temporal_files[0], str(os.path.join(data_directory, '30m-built-prediction.tif')),
                   np.asarray(Y_pred, dtype='uint8').reshape(IMG_SIZE, IMG_SIZE), 'uint8', 1)
    create_new_tif(temporal_files[0], str(os.path.join(data_directory, '30m-built-confidence.tif')),
                   np.asarray(class_0_probs, dtype='uint8').reshape(IMG_SIZE, IMG_SIZE), 'uint8', 1)

    vrt_file = configJson['commons']['original-built']
    clip_from_vrt(temporal_files[0], vrt_file, os.path.join(
        data_directory, 'trash.tif'), os.path.join(data_directory, '30m-original-built.tif'))

    tif2 = gdal.Open(os.path.join(data_directory, '30m-original-built.tif'))
    Y_raster = tif2.GetRasterBand(1).ReadAsArray()
    print(Y_raster.shape)
    Y_raster[Y_raster == 0] = 125
    Y_raster[Y_raster == 1] = 255
    Y_raster[Y_raster == 2] = 125
    Y_raster[Y_raster == 3] = 0
    Y_raster[Y_raster == 4] = 0
    Y_raster[Y_raster == 5] = 0
    Y_raster[Y_raster == 6] = 0
    Y_raster[Y_raster == 125] = 1
    Y_raster[Y_raster == 255] = 1
    print("UNIQUE CLASSES: ", np.unique(Y_raster))
    Y_list = Y_raster.flatten().tolist()
    Y_ground_truth = np.reshape(Y_list, (IMG_SIZE*IMG_SIZE, 1))
    print(Y_ground_truth.shape)

    dictionary = classification_report(
        Y_ground_truth, Y_pred, output_dict=True)
    print(dictionary)
    intersection = np.logical_and(Y_ground_truth, Y_pred)
    union = np.logical_or(Y_ground_truth, Y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    dictionary['meanIOU'] = iou_score
    with open(os.path.join(data_directory, 'metrics.json'), 'w') as fp:
        json.dump(dictionary, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
