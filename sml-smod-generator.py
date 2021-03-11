
### Author : Tanmay ###

import argparse
import codecs
import json
import math
import os
import subprocess

from tqdm import tqdm

import gdal
import numpy as np
import rasterio

# SEARCH AROUND A PIXEL


def classify_urban_area(i, j, d, arr):
    n = arr[i-d:i+d+1, j-d:j+d+1].flatten()
    return n

# CREATE A NEW TIF WITH GEO REFERENCE


def create_new_tif(sourceraster, targetraster, array2d, dtype, nbands):
    with rasterio.open(sourceraster) as src:
        metadata = src.profile
    metadata['count'] = nbands
    metadata['dtype'] = dtype
    with rasterio.open(targetraster, 'w', **metadata) as dst:
        dst.write(array2d, 1)
        print('New tif created at: ', str(targetraster))

# MAIN FUNCTION


def main():
    parser = argparse.ArgumentParser(
        description="CREATES SMOD raster from given BUILT (confidence) and POP rasters")
    parser.add_argument(
        "-cf", "--confidenceFile", help="Full file path of Built(Confidence) raster TIF file ")

    parser.add_argument(
        "-pr", "--predictionFile", help="Full file path of Built(Prediction) raster TIF file ")
    parser.add_argument("-p", "--popFile",
                        help="Full file path of POP raster TIF file ")
    parser.add_argument("-of", "--outputFolder",
                        help="Full desired output folder path")
    args = parser.parse_args()
    built_file = args.confidenceFile
    prediction_file = args.predictionFile
    pop_file = args.popFile
    output_file_path = args.outputFolder

    # VALIDATE ARGUMENTS
    if(built_file is None or pop_file is None or prediction_file is None):
        print("BUILT (confidence) or prediction or POP file is missing")
        exit()

    if(output_file_path is None):
        print("Output path is missing")
        exit()

    # READ META DATA OF FILES
    with rasterio.open(built_file.strip()) as src_built:
        prof_built = src_built.profile
        x_size_built = prof_built['width']
        y_size_built = prof_built['height']

        if(x_size_built != y_size_built):
            print('Please input n x n GHS-BUILT tif files only')
            exit()

    with rasterio.open(pop_file.strip()) as src_pop:
        prof_built = src_pop.profile
        x_size_pop = prof_built['width']
        y_size_pop = prof_built['height']

        if(x_size_pop != y_size_pop):
            print('Please input n x n GHS-POP tif files only')
            exit()

    if(x_size_built != x_size_pop):
        print('Input BUILT and POP files must be of same size')
        exit()

    # READ OUTPUT FILE PATHS
    current_file_name = os.path.splitext(os.path.basename(pop_file))[0].replace(
        '-confidence', '').replace('-built', '').replace('-pop', '')
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # SMOD ALGORITHM
    cc = gdal.Open(built_file)
    confidence2d = cc.GetRasterBand(1).ReadAsArray()
    dd = gdal.Open(pop_file)
    final_ghs_pop2d = dd.GetRasterBand(1).ReadAsArray()

    pp = gdal.Open(prediction_file)
    prediction2d = pp.GetRasterBand(1).ReadAsArray()

    urban_array = np.zeros(final_ghs_pop2d.shape, dtype='uint8')
    for ix, iy in tqdm(np.ndindex(final_ghs_pop2d.shape)):
        popcellvalue = final_ghs_pop2d[ix, iy]
        confidencecellvalue = confidence2d[ix, iy]
        predictioncellvalue = prediction2d[ix, iy]

        if((popcellvalue > 1500 or confidencecellvalue > 75) or (np.sum(classify_urban_area(ix, iy, 4, final_ghs_pop2d)) > 50000) and predictioncellvalue == 0):
            urban_array[ix, iy] = 3
        elif ((popcellvalue > 300 and confidencecellvalue > 60)and (np.sum(classify_urban_area(ix, iy, 1, final_ghs_pop2d)) > 5000)):
            urban_array[ix, iy] = 2
        elif(popcellvalue > 1 and (np.sum(classify_urban_area(ix, iy, 2, final_ghs_pop2d)) < 5000) and predictioncellvalue == 0):
            urban_array[ix, iy] = 1

    print(np.unique(urban_array, return_counts=True))

    # CREATE RASTER FILE OF GHS-SMOD PREDICTIONS
    create_new_tif(built_file.strip(), str(os.path.join(output_file_path, current_file_name+'-smod.tif')),
                   np.asarray(urban_array, dtype='uint8').reshape(x_size_pop, x_size_pop), 'uint8', 1)


if __name__ == '__main__':
    main()
