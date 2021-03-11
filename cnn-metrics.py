
import argparse
import gdal
import rasterio
from sklearn.metrics import classification_report
import json
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Calculates metrics between BUILT and Original GHSL-BUILT")
    parser.add_argument("-pr", "--prediction",
                        help="Full file path of BUILT prediction")
    parser.add_argument("-og", "--original",
                        help="Full file path of original GHSL BUILT")
    parser.add_argument("-of", "--outputFolder", help="Output folder path")
    args = parser.parse_args()
    prediction_file = args.prediction
    original_file = args.original
    output_file_path = args.outputFolder

    if (prediction_file is None or prediction_file.isspace()):
        print('BUILT prediction file is missing')
        exit()
    if (original_file is None or original_file.isspace()):
        print('Original GHSL BUILT prediction file is missing')
        exit()
    if (output_file_path is None or output_file_path.isspace()):
        print('Output folder is missing')
        exit()

        # READ META DATA OF INPUT FILE
    with rasterio.open(prediction_file.strip()) as src1:
        prof = src1.profile
        x_size_pred = prof['width']
        y_size_pred = prof['height']
        if(x_size_pred != y_size_pred):
            print('Please input n x n tif files only')
            exit()

        # READ META DATA OF INPUT FILE
    with rasterio.open(original_file.strip()) as src2:
        prof = src2.profile
        x_size_org = prof['width']
        y_size_org = prof['height']
        if(x_size_org != y_size_org):
            print('Please input n x n tif files only')
            exit()

    if(x_size_pred != x_size_org):
        print('BUILT file and Original GHSL BUILT file sizes do not match')
        exit()

    built_org1 = gdal.Open(original_file)
    built_org = built_org1.GetRasterBand(1).ReadAsArray()
    built_org[built_org == 0] = 125
    built_org[built_org == 1] = 255
    built_org[built_org == 2] = 125
    built_org[built_org == 3] = 0
    built_org[built_org == 4] = 0
    built_org[built_org == 5] = 0
    built_org[built_org == 6] = 0
    built_org[built_org == 125] = 1
    built_org[built_org == 255] = 1

    built_sml1 = gdal.Open(prediction_file)
    built_sml = built_sml1.GetRasterBand(1).ReadAsArray()
    built_sml = 1-built_sml
    dictionary = classification_report(
        built_org.flatten(), built_sml.flatten(), output_dict=True)
    intersection = np.logical_and(built_org.flatten(), built_sml.flatten())
    union = np.logical_or(built_org.flatten(), built_sml.flatten())
    iou_score = np.sum(intersection) / np.sum(union)
    dictionary['meanIOU'] = iou_score
    with open(os.path.join(output_file_path, 'metrics.json'), 'w') as fp:
        json.dump(dictionary, fp, indent=4, sort_keys=True)



if __name__ == '__main__':
    main()
