### Author : Tanmay ###


import argparse
import codecs
import json
import math
import os
import pickle
import subprocess
from collections import Counter

from tqdm import tqdm

import gdal
import numpy as np
import rasterio
from sklearn.preprocessing import minmax_scale


# READ N-BANDS from MULTISPECTRAL(MS) RASTER FILE
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


# READ A PIXEL SEQUENCE CORRESPONDING TO N-BANDS IN THE SAME IMAGE
def data_point_bands_list(dataset, index, N_BANDS):
    temp_list = []
    for b in range(N_BANDS):
        temp_list.append(dataset[b][index])

    return temp_list

# SCALE VALUES FROM  [-1,-] TO [0,100]


def scale_value(input_value):
    return ((((input_value - (-1)) * (100 - 0)) / (1 - (-1))) + 0)


def change_pixel_size(inputfile, outputfile, pixelsize):
    subprocess.call('gdalwarp -tr ' + str(pixelsize)+' ' +
                    str(pixelsize)+' '+inputfile+' '+outputfile, shell=True)

# CREATE NEW RASTER WITH GEOREFERENCE


def create_new_tif(sourceraster, targetraster, array2d, dtype, nbands):
    with rasterio.open(sourceraster) as src:
        metadata = src.profile
    metadata['count'] = nbands
    metadata['dtype'] = dtype
    with rasterio.open(targetraster, 'w', **metadata) as dst:
        dst.write(array2d, 1)
        print('New tif created at: ', str(targetraster))

# CLASSIFICATION OF A RASTER IMAGE FROM PRE-TRAINED N-BAND MODEL


def classification(file_name, model_name, N_BANDS, THRESHOLD):

    test_data_all_bands = read_all_bands(file_name, N_BANDS)

    # LOAD DICTIONARY OF ENDI MEASURE
    ENDI_table_dictionary_model = pickle.load(open(model_name, 'rb'))

    for i in range(0, N_BANDS):
        # Q value - QUANTAIZATION
        q = max(test_data_all_bands[i])/16
        test_data_all_bands[i] = [round(math.floor(x+0.5)/q)
                                  for x in test_data_all_bands[i]]

    final_pred_list = []
    final_confidence_list = []
    final_endi_list = []
    print(f'Using BANDS: {N_BANDS} and THRESHOLD: {THRESHOLD}')

    for i in tqdm(range(len(test_data_all_bands[0]))):

        data = data_point_bands_list(test_data_all_bands, i, N_BANDS)
        tup = tuple(data)

        if tup in ENDI_table_dictionary_model:
            ENDI_vals = ENDI_table_dictionary_model[tup]
            max_val = max(ENDI_vals)
            if max_val >= THRESHOLD:
                index_max = ENDI_vals.index(max_val)
                final_pred_list.append(index_max)
                final_confidence_list.append(scale_value(ENDI_vals[0]))
                final_endi_list.append(ENDI_vals[0])
            else:
                final_pred_list.append(1)
                final_confidence_list.append(scale_value(ENDI_vals[0]))
                final_endi_list.append(ENDI_vals[0])

        else:
            final_pred_list.append(1)
            final_confidence_list.append(scale_value(-1))
            final_endi_list.append(-1)

    return final_pred_list, final_confidence_list, final_endi_list


def main():
    parser = argparse.ArgumentParser(
        description="CREATES BUILT (0/1) and BUILT (confidence) rasters")
    parser.add_argument("-c", "--config", help="Full file path of config.json")
    parser.add_argument(
        "-nb", "--bands", help="No. of bands to be used in inference")
    parser.add_argument(
        "-i", "--inputFile", help="Full file path of multispectral(MS) satellite image with stacked N-BANDS")
    parser.add_argument("-of", "--outputFolder",
                        help="Full desired output folder path")
    parser.add_argument("-th", "--threshold",
                        help="Threshold to control predictions")
    args = parser.parse_args()
    config_file_path = args.config
    model_bands = args.bands
    input_file = args.inputFile
    output_file_path = args.outputFolder
    threshold = args.threshold

    # VALIDATE ARGUMENTS
    if(config_file_path is None or config_file_path.isspace()):
        print("Config file path missing, please use '-c' option")
        exit()

    if(model_bands is None or model_bands.isspace()):
        print("N-band model is missing, input number of bands, please use '-nb' option")
        exit()

    if(input_file is None or input_file.isspace()):
        print("Multispectral input TIF file is missing")
        exit()

    if(output_file_path is None or output_file_path.isspace()):
        print("Output path is missing")
        exit()

    if(threshold is None or threshold.isspace()):
        threshold = '0.02'

    # READ META DATA OF INPUT FILE
    with rasterio.open(input_file.strip()) as src:
        prof = src.profile
        x_size = prof['width']
        y_size = prof['height']
        if(x_size != y_size):
            print('Please input n x n tif files only')
            exit()

    # READ CONFIG JSON
    configJson = json.load(codecs.open(config_file_path, 'r', 'utf-8-sig'))

    N_BANDS = int(model_bands)
    model_key = str(model_bands)+'-band'
    model = configJson['sml-built']['test']['models'][model_key]

    prediction_list, confidence_list, endi_list = classification(
        input_file.strip(), model.strip(), N_BANDS, float(threshold))

    # READ OUTPUT FILE PATHS
    current_file_name = os.path.splitext(os.path.basename(input_file))[
        0]+'-'+str(N_BANDS)
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # CREATE RASTER FILES OF GHS-BUILT PREDICTIONS AND CONFIDENCE
    create_new_tif(input_file.strip(), str(os.path.join(output_file_path, current_file_name+'-built-prediction.tif')),
                   np.asarray(prediction_list, dtype='uint8').reshape(x_size, x_size), 'uint8', 1)
    create_new_tif(input_file.strip(), str(os.path.join(output_file_path, current_file_name+'-built-confidence.tif')),
                   np.asarray(confidence_list, dtype='uint8').reshape(x_size, x_size), 'uint8', 1)


if __name__ == '__main__':
    main()
