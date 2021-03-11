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
from rasterio.coords import disjoint_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from sklearn.preprocessing import minmax_scale

# RESIZE A RASTER FILE


def resize_scene_custom_size(input_file, output_file, xsize, ysize):
    subprocess.call('gdal_translate -co QUALITY=100 -co PROGRESSIVE=ON -outsize ' +
                    str(xsize)+' '+str(ysize)+' '+input_file+' '+output_file, shell=True)

# CREATE A NEW RASTER FILE WITH GEO REFERENCE


def create_new_tif(sourceraster, targetraster, array2d, dtype, nbands):
    with rasterio.open(sourceraster) as src:
        metadata = src.profile
    metadata['count'] = nbands
    metadata['dtype'] = dtype
    with rasterio.open(targetraster, 'w', **metadata) as dst:
        dst.write(array2d, 1)
        print('New tif created at: ', str(targetraster))


# CLIP SMALL TIF IMAGE FROM A LARGER IMAGE WITH OVERLAPPING EXTENSIONS
def clip_from_template(input, like, output):
    with rasterio.open(input) as src:
        with rasterio.open(like) as template_ds:
            bounds = template_ds.bounds
            if template_ds.crs != src.crs:
                bounds = transform_bounds(template_ds.crs, src.crs,
                                          *bounds)

            if disjoint_bounds(bounds, src.bounds):
                print('must overlap the extent of '
                      'the input raster')
                exit()

        bounds_window = src.window(*bounds)
        bounds_window = bounds_window.intersection(
            Window(0, 0, src.width, src.height))

        out_window = bounds_window.round_lengths(op='ceil')

        height = int(out_window.height)
        width = int(out_window.width)

        out_kwargs = src.profile
        out_kwargs.update({
            # 'driver': driver,
            'height': height,
            'width': width,
            'transform': src.window_transform(out_window)})

        with rasterio.open(output, 'w', **out_kwargs) as out:
            out.write(src.read(window=out_window,
                               out_shape=(src.count, height, width)))


# MAIN FUNCTION
def main():
    parser = argparse.ArgumentParser(
        description="CREATES GHS-POP from GHS-BUILT (confidence) and GPW data")
    parser.add_argument("-c", "--config", help="Full file path of config.json")
    parser.add_argument(
        "-cf", "--inputFile", help="Full file path of GHS-Built(Confidence) raster TIF file ")
    parser.add_argument(
        "-pr", "--predictionFile", help="Full file path of GHS-Built(Prediction) raster TIF file ")
    parser.add_argument("-of", "--outputFolder",
                        help="Full desired output folder path")
    args = parser.parse_args()
    config_file_path = args.config
    input_file = args.inputFile
    prediction_file = args.predictionFile
    output_file_path = args.outputFolder

    # VALIDATE ARGUMENTS
    if(config_file_path is None):
        print("Config file path missing, please use '-c' option")
        exit()

    if(input_file is None or prediction_file is None):
        print("GHS-BUILT (confidence) or (prediction) input TIF file is missing")
        exit()

    if(output_file_path is None):
        print("Output path is missing")
        exit()

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

    # READ OUTPUT FILE PATHS
    current_file_name = os.path.splitext(os.path.basename(input_file))[
        0].replace('-built', '').replace('-confidence', '')
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    temp_pop_clip_path = os.path.join(output_file_path, 'temp-pop-clips')
    if not os.path.exists(temp_pop_clip_path):
        os.makedirs(temp_pop_clip_path)

    gpw = configJson['sml-pop']['gpw-count-path']

    temp_pop_file = os.path.join(
        temp_pop_clip_path, current_file_name+'-temp-pop.tif')
    clip_from_template(gpw.strip(), input_file.strip(), temp_pop_file)
    temp_pop_file_resized = os.path.join(
        temp_pop_clip_path, current_file_name+'-temp-resized-pop.tif')
    resize_scene_custom_size(
        temp_pop_file, temp_pop_file_resized, x_size, x_size)

    # POP GENERATION WITH DASYMMETRIC ALGORITHM
    pop1 = gdal.Open(temp_pop_file_resized)
    pop2d = pop1.GetRasterBand(1).ReadAsArray()
    pop2d[pop2d < 0] = 0
    pop1d = pop2d.flatten()
    print('Highest population: ', np.max(pop1d))
    print('Least population: ', np.min(pop1d))

    # CONFIDENCE VALUE CATEGORIZATION
    a = gdal.Open(input_file)
    confidence_values_category = a.GetRasterBand(1).ReadAsArray().flatten()

    b = gdal.Open(prediction_file)
    prediction_values_category = b.GetRasterBand(1).ReadAsArray().flatten()

    a = None
    b = None

    confidence_values_category[confidence_values_category < 40] = 0
    confidence_values_category[(confidence_values_category < 45) & (
        confidence_values_category >= 40)] = 1
    confidence_values_category[(confidence_values_category < 50) & (
        confidence_values_category >= 45)] = 2
    confidence_values_category[(confidence_values_category < 55) & (
        confidence_values_category >= 50)] = 3
    confidence_values_category[(confidence_values_category < 60) & (
        confidence_values_category >= 55)] = 4
    confidence_values_category[(confidence_values_category < 65) & (
        confidence_values_category >= 60)] = 5
    confidence_values_category[confidence_values_category >= 65] = 6

    for i in tqdm(range(prediction_values_category.size)):
        if(prediction_values_category[i]) == 2:
            confidence_values_category[i] = 0

    eachcount_confidence = []
    for i in range(0, 7):
        eachcount_confidence.append((confidence_values_category == i).sum())

    population = [0, 0, 0, 0, 0, 0, 0]

    for i in tqdm(range(0, pop1d.size)):
        if(confidence_values_category[i] == 1):
            population[1] = population[1]+pop1d[i]
        elif(confidence_values_category[i] == 2):
            population[2] = population[2]+pop1d[i]
        elif(confidence_values_category[i] == 3):
            population[3] = population[3]+pop1d[i]
        elif(confidence_values_category[i] == 4):
            population[4] = population[4]+pop1d[i]
        elif(confidence_values_category[i] == 5):
            population[5] = population[5]+pop1d[i]
        elif(confidence_values_category[i] == 6):
            population[6] = population[6]+pop1d[i]

    area = []
    for i in tqdm(range(0, 7)):
        area.append(eachcount_confidence[i]*10*10)

    density = []
    for i in range(0, 7):
        density.append(population[i]/area[i])

    total_density_denominator = np.nansum(density)
    d_values = []
    for i in range(0, 7):
        d_values.append(density[i]/total_density_denominator)

    pop1d_size = pop1d.size
    a_values = []
    for i in range(0, 7):
        a_values.append((eachcount_confidence[i]/pop1d_size)/0.33)

    d_into_a_values = []
    for i in range(0, 7):
        d_into_a_values.append(d_values[i]*a_values[i])

    f_values = []
    d_into_a_values_total_denominator = np.nansum(d_into_a_values)
    for i in range(0, 7):
        f_values.append(d_into_a_values[i]/d_into_a_values_total_denominator)

    total_pop = np.nansum(pop1d)
    pop_values = []
    for i in range(0, 7):
        pop_values.append((f_values[i]*total_pop)/eachcount_confidence[i])

    final_ghspop1d = []
    for i in tqdm(range(0, pop1d.size)):
        if(confidence_values_category[i] == 1):
            final_ghspop1d.append(pop_values[1])
        elif(confidence_values_category[i] == 2):
            final_ghspop1d.append(pop_values[2])
        elif(confidence_values_category[i] == 3):
            final_ghspop1d.append(pop_values[3])
        elif(confidence_values_category[i] == 4):
            final_ghspop1d.append(pop_values[4])
        elif(confidence_values_category[i] == 5):
            final_ghspop1d.append(pop_values[5])
        elif(confidence_values_category[i] == 6):
            final_ghspop1d.append(pop_values[6])
        else:
            final_ghspop1d.append(pop_values[0])

    # CREATE RASTER FILES OF GHS-POP PREDICTIONS
    create_new_tif(input_file.strip(), str(os.path.join(output_file_path, current_file_name+'-pop.tif')),
                   np.asarray(final_ghspop1d, dtype='float32').reshape(x_size, x_size), 'float32', 1)


if __name__ == '__main__':
    main()
