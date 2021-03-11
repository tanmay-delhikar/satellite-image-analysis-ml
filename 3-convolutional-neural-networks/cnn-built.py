### AUTHOR : Tanmay ###

import argparse
import os
import subprocess
import codecs
import json
from skimage.external import tifffile
import rasterio

# CONVERT JP2 IMAGE TO TIFF


def jp2_to_tif(input_file, output_file):
    subprocess.call('gdal_translate -tr 10 10 ' +
                    input_file+' '+output_file, shell=True)

# SORT SENTINEL2 BANDS IN ORDER


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

# STACK N BANDS IN SINGLE FILE


def stack_bands(file_list, output_file):
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    meta.update(count=len(file_list))

    with rasterio.open(output_file, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


# CREATE MULTISPECTRAL  IMAGE
def ms_creator(safe_folder_path, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tif_files_temp = []
    for i in os.listdir(safe_folder_path):
        if i.endswith('.tif') and '_B' in i:
            tif_files_temp.append(os.path.join(safe_folder_path, i))
    tif_files_temp.sort()
    tif_files = band_sorter_S2(tif_files_temp)
    print('Sortedbands: ', str(tif_files))

    output_file = os.path.join(output_folder, 'ms.tif')
    stack_bands(tif_files, output_file)
    return output_file


def remove_bad_size(outpath, size):
    for tile in os.listdir(outpath):
        tileFile = os.path.join(outpath, tile)
        im = tifffile.imread(tileFile)
        if im.shape[0] != size or im.shape[1] != size:
            os.remove(tileFile)
            print("removed: ", tile)


def tile_creator(rgb_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    subprocess.call('gdal_retile.py -targetDir ' +
                    output_folder + ' -ps 256 256 ' + rgb_file, shell=True)
    remove_bad_size(output_folder, 256)


def preprocess_s2(input_safe_path, output_safe_path):

    if not os.path.exists(output_safe_path):
        os.makedirs(output_safe_path)

    for root, dir, filelist in os.walk(input_safe_path):
        if root.endswith('IMG_DATA'):
            img_data_path = root

    for i in os.listdir(img_data_path):
        if i.endswith('.jp2') and '_B' in i:
            print('Processing...'+i)
            jp2_to_tif(os.path.join(img_data_path, i), os.path.join(
                output_safe_path, os.path.splitext(i)[0]+'.tif'))


def ghsl_all_products_cnn(safe_folder_path, output_folder_path, inference_conf_file, inference_pred_file, cnn_config):

    OUTPUT_FOLDER = os.path.join(output_folder_path, 'confidence')

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    preprocess_s2(safe_folder_path, OUTPUT_FOLDER)

    ms_file = ms_creator(OUTPUT_FOLDER, OUTPUT_FOLDER)
    OUTPUT_FOLDER_TILES = os.path.join(OUTPUT_FOLDER, 'tiles')
    OUTPUT_FOLDER_TILES_T = os.path.join(OUTPUT_FOLDER_TILES, 't')
    tile_creator(ms_file, OUTPUT_FOLDER_TILES_T)
    subprocess.call('python3 '+inference_conf_file+' -c '+cnn_config +
                    ' -d '+OUTPUT_FOLDER_TILES+' -o '+OUTPUT_FOLDER, shell=True)
    OUTPUT_FOLDER = os.path.join(output_folder_path, 'prediction')

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    preprocess_s2(safe_folder_path, OUTPUT_FOLDER)
    ms_file = ms_creator(OUTPUT_FOLDER, OUTPUT_FOLDER)
    OUTPUT_FOLDER_TILES = os.path.join(OUTPUT_FOLDER, 'tiles')
    OUTPUT_FOLDER_TILES_T = os.path.join(OUTPUT_FOLDER_TILES, 't')
    tile_creator(ms_file, OUTPUT_FOLDER_TILES_T)
    subprocess.call('python3 '+inference_pred_file+' -c '+cnn_config +
                    ' -d '+OUTPUT_FOLDER_TILES+' -o '+OUTPUT_FOLDER, shell=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CNN based BUILT UP(0/1) and CONFIDENCE")
    parser.add_argument("-c", "--configJson", help="Config json path")
    parser.add_argument("-if", "--inputFolder",
                        help="Root folder containing list of tif files")
    parser.add_argument("-of", "--outputFolder",
                        help="Output path to save files")

    args = parser.parse_args()
    config_json_path = args.configJson
    input_folder = args.inputFolder
    output_folder = args.outputFolder

    if(input_folder is None or output_folder is None or config_json_path is None):
        print("Either of config json, input folder, output folder is missing")
        exit()

    # READ CONFIG JSON
    configJson = json.load(codecs.open(config_json_path, 'r', 'utf-8-sig'))

    inference_conf_file = configJson['cnn']['inference-conf']
    inference_pred_file = configJson['cnn']['inference-pred']
    cnn_config = configJson['cnn']['config']
    ghsl_all_products_cnn(input_folder, output_folder,
                          inference_conf_file, inference_pred_file, cnn_config)


if __name__ == '__main__':
    main()
