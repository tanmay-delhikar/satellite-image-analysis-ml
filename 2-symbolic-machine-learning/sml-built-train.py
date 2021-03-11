# TRAIN SYMBOLIC MACHINE LEARNING ALGORITHM
import argparse
import codecs
import json
import math
import os
import pickle
import subprocess
from collections import Counter

from sentinelsat.sentinel import SentinelAPI, geojson_to_wkt, read_geojson
from tqdm import tqdm

import cv2
import gdal
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import rasterio.merge
import rasterio.warp
from gdalconst import GA_ReadOnly
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.coords import disjoint_bounds
from rasterio.plot import plotting_extent, show
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from sklearn.preprocessing import minmax_scale

bounds = None
res = None
nodata = None
precision = 7


# HELPER FUNCTIONS
def resize_scene_custom_size(input_file, output_file, xsize, ysize):
    subprocess.call('gdal_translate -co QUALITY=100 -co PROGRESSIVE=ON -outsize ' +
                    str(xsize)+' '+str(ysize)+' '+input_file+' '+output_file, shell=True)


def change_pixel_size(inputfile, outputfile, pixelsize):
    subprocess.call('gdalwarp -tr ' + str(pixelsize)+' ' +
                    str(pixelsize)+' '+inputfile+' '+outputfile, shell=True)


def read_all_bands(file_name, N_BANDS):
    bands_data = []
    print('Reading tif file: ', file_name)
    tif = gdal.Open(file_name)
    for i in range(1, N_BANDS+1):
        print('Reading band data: ', i)
        temp_data = tif.GetRasterBand(i).ReadAsArray()
        print(temp_data.shape)
        bands_data.append(temp_data.flatten().tolist())

    return bands_data


def data_point_bands_list(dataset, index, N_BANDS):
    temp_list = []
    for b in range(N_BANDS):
        temp_list.append(dataset[b][index])

    return temp_list


def calculate_ENDI(positive, negative):
    if (positive == 0) and (negative == 0):
        return 0
    return (positive-negative)/(positive+negative)


# POSITIVE EVIDENCES
def positive_count(sequence, class_name, counter):
    item = list(sequence)
    item.append(class_name)
    c = counter[tuple(item)]
    return c

# NEGATIVE EVIDENCES


def negative_count(sequence, class_list, counter):
    negative_classcount = 0
    for c in class_list:
        item = list(sequence)
        item.append(c)
        negative_classcount = negative_classcount+counter[tuple(item)]

    return negative_classcount


def main():
    parser = argparse.ArgumentParser(
        description="TRAINS SML ALGORITHM")
    parser.add_argument("-c", "--config", help="Full file path of config.json")
    parser.add_argument(
        "-nb", "--bands", help="No. of bands to be used in training")

    args = parser.parse_args()
    config_file_path = args.config
    model_bands = args.bands

    # VALIDATE ARGUMENTS
    if(config_file_path is None):
        print("Config file path missing, please use '-c' option")
        exit()

    if(model_bands is None):
        print("N-band model is missing, input number of bands, please use '-nb' option")
        exit()

        # READ CONFIG JSON
    configJson = json.load(codecs.open(config_file_path, 'r', 'utf-8-sig'))

    N_BANDS = int(model_bands)
    output_path = configJson['sml-built']['train']['output-path']
    train_path = configJson['sml-built']['train']['input-path']

    TRAIN_DATA_DIRECTORY_X = os.path.join(train_path, 'X')
    TRAIN_DATA_DIRECTORY_Y = os.path.join(train_path, 'Y')

    all_files = os.listdir(TRAIN_DATA_DIRECTORY_X)
    all_files.sort()
    print('Files in training directory: ', all_files)

    all_files_data = []
    all_files_ref_data = []

    for a_file in all_files:
        if '.tif' not in a_file:
            continue

        # train data
        all_files_data.append(read_all_bands(
            os.path.join(TRAIN_DATA_DIRECTORY_X, a_file), N_BANDS))

        # reference data
        ref_tif = gdal.Open(os.path.join(TRAIN_DATA_DIRECTORY_Y, a_file))
        temp_data = ref_tif.GetRasterBand(1).ReadAsArray()
        temp_data[temp_data == 0] = 125
        temp_data[temp_data == 1] = 255
        temp_data[temp_data == 2] = 125
        temp_data[temp_data == 3] = 0
        temp_data[temp_data == 4] = 0
        temp_data[temp_data == 5] = 0
        temp_data[temp_data == 6] = 0
        all_files_ref_data = all_files_ref_data+(temp_data.flatten().tolist())

    all_bands_data = []
    for i in range(N_BANDS):
        all_bands_data.append([])
    for i in range(len(all_files_data)):
        for j in range(N_BANDS):
            all_bands_data[j] = all_bands_data[j]+all_files_data[i][j]

    all_files_data = None

    for i in range(0, N_BANDS):
        # Q value
        q = max(all_bands_data[i])/16
        all_bands_data[i] = [round(math.floor(x+0.5)/q)
                             for x in all_bands_data[i]]

    ante_consq_list = []
    sequences_list = []

    for i in tqdm(range(len(all_bands_data[0]))):

        data = data_point_bands_list(all_bands_data, i, N_BANDS)
        sequences_list.append(tuple(data))

        data.append(all_files_ref_data[i])
        ante_consq_list.append(tuple(data))

    print('UNIQUE SEQUENCES COUNT: ')
    print(len(list(set(sequences_list))))
    unique_X_sequences = list(set(sequences_list))

    # COUNT FREQUENCIES OF EACH UNIQUE SEQUENCE
    counter = Counter(ante_consq_list)
    class_uniques, class_counts = np.unique(
        all_files_ref_data, return_counts=True)
    reference_data_size = len(all_files_ref_data)

    # GENERATE A DICTIONARY OF THE FORM : {(unique sequence) : [ENDI MEASURE-CLASS 0, ENDI MEASURE-CLASS 1]}
    ENDI_table_dictionary = {}
    for i in tqdm(range(len(unique_X_sequences))):
        seq = unique_X_sequences[i]
        ENDI_class_0_a = calculate_ENDI(positive_count(
            seq, 0, counter), negative_count(seq, [125, 255], counter))
        ENDI_class_0_b = calculate_ENDI(positive_count(seq, 0, counter)/class_counts[0], negative_count(
            seq, [125, 255], counter)/(reference_data_size-class_counts[0]))
        ENDI_class_0_ab = (ENDI_class_0_a+ENDI_class_0_b)/2

        ENDI_class_1_a = calculate_ENDI(positive_count(
            seq, 125, counter), negative_count(seq, [0, 255], counter))
        ENDI_class_1_b = calculate_ENDI(positive_count(
            seq, 125, counter)/class_counts[1], negative_count(seq, [0, 255], counter)/(reference_data_size-class_counts[1]))
        ENDI_class_1_ab = (ENDI_class_1_a+ENDI_class_1_b)/2

        ENDI_class_2_a = calculate_ENDI(positive_count(
            seq, 255, counter), negative_count(seq, [0, 125], counter))
        ENDI_class_2_b = calculate_ENDI(positive_count(
            seq, 255, counter)/class_counts[2], negative_count(seq, [0, 125], counter)/(reference_data_size-class_counts[2]))
        ENDI_class_2_ab = (ENDI_class_2_a+ENDI_class_2_b)/2

        ENDI_table_dictionary[seq] = [
            ENDI_class_0_ab, ENDI_class_1_ab, ENDI_class_2_ab]

    print('BUILDING DICTIONARY DONE')
    output_model_full_path = os.path.join(
        output_path, 'sml-model-'+str(N_BANDS)+'-bands'+'.pkl')
    pickle.dump(ENDI_table_dictionary, open(output_model_full_path, 'wb'))
    print('Model dumped in: ', output_model_full_path)



if __name__ == '__main__':
    main()
