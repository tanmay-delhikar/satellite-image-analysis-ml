import argparse
import multiprocessing
import os
import subprocess

import rasterio


def resize_scene_custom_size(input_file, output_file, xsize, ysize):
    subprocess.call('gdal_translate -co QUALITY=100 -co PROGRESSIVE=ON -outsize ' +
                    str(xsize)+' '+str(ysize)+' '+input_file+' '+output_file, shell=True)


def generate_wsf_all_products(input_folder, output_folder_path, config_json_path):

    WSF_OUTPUT_PATH = os.path.join(output_folder_path, 'WSF')
    if not os.path.exists(WSF_OUTPUT_PATH):
        os.makedirs(WSF_OUTPUT_PATH)

    subprocess.call('python3 wsf-built.py -if '+input_folder +
                    ' -of '+output_folder_path+' -c '+config_json_path, shell=True)

    PREDICTION_FILE_PATH = os.path.join(
        WSF_OUTPUT_PATH, '10m-built-prediction.tif')
    with rasterio.open(PREDICTION_FILE_PATH) as src:
        prof = src.profile
        x_size = prof['width']
        y_size = prof['height']
        if x_size != y_size:
            print('Input must be of only n x n size')
            exit()

    CONFIDENCE_FILE_PATH = os.path.join(
        WSF_OUTPUT_PATH, '10m-built-confidence.tif')
    POP_FILE = os.path.join(WSF_OUTPUT_PATH, '10m-pop.tif')
    subprocess.call('python3 wsf-spop-gpw-count-generator.py -c '+config_json_path+' -cf ' +
                    CONFIDENCE_FILE_PATH+' -pr '+PREDICTION_FILE_PATH+' -of '+WSF_OUTPUT_PATH, shell=True)
    subprocess.call('python3 wsf-smod-generator.py '+' -cf '+CONFIDENCE_FILE_PATH +
                    ' -pr '+PREDICTION_FILE_PATH+' -p '+POP_FILE+' -of '+WSF_OUTPUT_PATH, shell=True)
    print('Done!')



def main():
    parser = argparse.ArgumentParser(
        description="Generates WSF based BUILT,POP,SMOD")
    parser.add_argument("-c", "--configJson", help="Config json path")
    parser.add_argument("-if", "--inputFolder",
                        help="Root folder containing list of .SAFE folders")
    parser.add_argument("-of", "--outputFolder",
                        help="Output path to save preprocessed files")
    parser.add_argument("-st", "--satelliteType",
                        help="Satellite type S2 OR L8")

    args = parser.parse_args()
    config_json_path = args.configJson
    root_foler = args.inputFolder
    output_folder = args.outputFolder
    st = args.satelliteType

    if(config_json_path is None or root_foler is None or output_folder is None):
        print("Either of config json, root folder, output folder is missing")
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    generate_wsf_all_products(root_foler, output_folder, config_json_path)

    print('All done!')


if __name__ == '__main__':
    main()
