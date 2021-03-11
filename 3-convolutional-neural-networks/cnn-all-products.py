### AUTHOR : Tanmay ###

# IMPORTS
import argparse
import multiprocessing
import os
import subprocess

import rasterio

# RESIZE IMAGE TO ANY SIZE


def resize_scene_custom_size(input_file, output_file, xsize, ysize):
    subprocess.call('gdal_translate -co QUALITY=100 -co PROGRESSIVE=ON -outsize ' +
                    str(xsize)+' '+str(ysize)+' '+input_file+' '+output_file, shell=True)


# GENERATE ALL CNN BASED BUILT->POP->SMOD PRODUCTS
def generate_cnn_all_products(safe_folder_path, output_folder_path, satellite_type, config_json_path):

    # CNN
    CNN_OUTPUT_PATH = os.path.join(output_folder_path, 'CNN')
    if not os.path.exists(CNN_OUTPUT_PATH):
        os.makedirs(CNN_OUTPUT_PATH)

    subprocess.call('python3 cnn-built.py -if '+safe_folder_path +
                    ' -of '+CNN_OUTPUT_PATH+' -c '+config_json_path, shell=True)

    PREDICTION_ROOT_PATH = os.path.join(CNN_OUTPUT_PATH, 'prediction')
    PREDICTION_SUB_FOLDER_PATH = os.path.join(
        PREDICTION_ROOT_PATH, 'U-Net_MS_hs_5_tiles')
    PREDICTION_FILE_PATH = os.path.join(
        PREDICTION_SUB_FOLDER_PATH, 'merged.tif')
    with rasterio.open(PREDICTION_FILE_PATH) as src:
        prof = src.profile
        x_size = prof['width']
        y_size = prof['height']
    BUILT_PREDICTION_FILE = os.path.join(
        CNN_OUTPUT_PATH, '10m-13-built-prediction.tif')
    resize_scene_custom_size(PREDICTION_FILE_PATH,
                             BUILT_PREDICTION_FILE, y_size, y_size)

    CONFIDENCE_ROOT_PATH = os.path.join(CNN_OUTPUT_PATH, 'confidence')
    CONFIDENCE_SUB_FOLDER_PATH = os.path.join(
        CONFIDENCE_ROOT_PATH, 'U-Net_MS_hs_5_tiles')
    CONFIDENCE_FILE_PATH = os.path.join(
        CONFIDENCE_SUB_FOLDER_PATH, 'merged.tif')
    BUILT_CONFIDENCE_FILE = os.path.join(
        CNN_OUTPUT_PATH, '10m-13-built-confidence.tif')
    resize_scene_custom_size(CONFIDENCE_FILE_PATH,
                             BUILT_CONFIDENCE_FILE, y_size, y_size)
    POP_FILE = os.path.join(CNN_OUTPUT_PATH, '10m-13-pop.tif')

    subprocess.call('python3 cnn-spop-gpw-count-generator.py -c '+config_json_path+' -cf ' +
                    BUILT_CONFIDENCE_FILE+' -pr '+BUILT_PREDICTION_FILE+' -of '+CNN_OUTPUT_PATH, shell=True)
    subprocess.call('python3 cnn-smod-generator.py '+' -cf '+BUILT_CONFIDENCE_FILE +
                    ' -pr '+BUILT_PREDICTION_FILE+' -p '+POP_FILE+' -of '+CNN_OUTPUT_PATH, shell=True)

    # ORGINAL REFERNCE FILES
    ORIGINAL_OUTPUT_PATH = os.path.join(output_folder_path, 'ORIGINAL')
    if not os.path.exists(ORIGINAL_OUTPUT_PATH):
        os.makedirs(ORIGINAL_OUTPUT_PATH)
    ORIGINAL_BUILT_PREDICTION_FILE = os.path.join(
        ORIGINAL_OUTPUT_PATH, '10m-original-built.tif')
    # METRICS
    subprocess.call('python3 cnn-metrics.py -pr '+BUILT_PREDICTION_FILE+' -og ' +
                    ORIGINAL_BUILT_PREDICTION_FILE+' -of '+CNN_OUTPUT_PATH, shell=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CNN based BUILT,POP,SMOD")
    parser.add_argument("-c", "--configJson", help="Config json path")
    parser.add_argument("-rf", "--rootFolder",
                        help="Root folder containing list of .SAFE folders")
    parser.add_argument("-of", "--outputFolder",
                        help="Output path to save processed files")
    parser.add_argument("-st", "--satelliteType",
                        help="Satellite type s2 OR l8")

    args = parser.parse_args()
    config_json_path = args.configJson
    root_foler = args.rootFolder
    output_folder = args.outputFolder
    st = args.satelliteType

    if(config_json_path is None or root_foler is None or output_folder is None):
        print("Either of config json, root folder, output folder is missing")
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    SAFE_FOLDERS_LIST = next(os.walk(root_foler))[1]
    SAFE_FOLDERS_LIST.sort()

    for safe_path in SAFE_FOLDERS_LIST:
        final_input_path = os.path.join(root_foler, safe_path)
        final_output_path = os.path.join(output_folder, safe_path)
        generate_cnn_all_products(
            final_input_path, final_output_path, st, config_json_path)

    print('All done!')


if __name__ == '__main__':
    main()
