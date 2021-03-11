
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
from gdalconst import GA_ReadOnly
from rasterio.coords import disjoint_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from sklearn.preprocessing import minmax_scale


def resize_scene_custom_size(input_file, output_file, xsize, ysize):
    subprocess.call('gdal_translate -co QUALITY=100 -co PROGRESSIVE=ON -outsize ' +
                    str(xsize)+' '+str(ysize)+' '+input_file+' '+output_file, shell=True)


def change_pixel_size(inputfile, outputfile, pixelsize):
    subprocess.call('gdalwarp -tr ' + str(pixelsize)+' ' +
                    str(pixelsize)+' '+inputfile+' '+outputfile, shell=True)


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

# MAIN FUNCTION


def main():
    parser = argparse.ArgumentParser(
        description="CLIPS FILES FROM ORIGINAL GHSL PRODUCTS")
    parser.add_argument("-c", "--config", help="Full file path of config.json")
    parser.add_argument(
        "-rf", "--referenceFile", help="Full file path of reference file to clip ")

    parser.add_argument("-of", "--outputFolder",
                        help="Full desired output folder path")
    args = parser.parse_args()
    config_file_path = args.config
    reference_file = args.referenceFile
    output_file_path = args.outputFolder

    # VALIDATE ARGUMENTS
    if(config_file_path is None):
        print("Config file path missing, please use '-c' option")
        exit()

    if(reference_file is None):
        print("Reference TIF file is missing")
        exit()

    if(output_file_path is None):
        print("Output path is missing")
        exit()

    # READ META DATA OF INPUT FILE
    with rasterio.open(reference_file.strip()) as src:
        prof = src.profile
        x_size = prof['width']
        y_size = prof['height']
        if(x_size != y_size):
            print('Please input n x n tif files only')
            exit()

    # READ CONFIG JSON
    configJson = json.load(codecs.open(config_file_path, 'r', 'utf-8-sig'))
    built = configJson['commons']['original-built']
    pop = configJson['commons']['original-spop']
    smod = configJson['commons']['original-smod']

    temp_clip_path = os.path.join(output_file_path, 'originals-temp')
    if not os.path.exists(temp_clip_path):
        os.makedirs(temp_clip_path)

    # GHS-BUILT
    temp_built_file = os.path.join(temp_clip_path, 'original-built.tif')
    temp_built_file2 = os.path.join(temp_clip_path, 'original-built-temp.tif')
    clip_from_vrt(reference_file.strip(), built,
                  temp_built_file2, temp_built_file)
    built_file_resized = os.path.join(
        output_file_path, '10m-original-built.tif')
    resize_scene_custom_size(
        temp_built_file, built_file_resized, x_size, x_size)
    built_file_resized_30m = os.path.join(
        output_file_path, '30m-original-built.tif')
    change_pixel_size(built_file_resized, built_file_resized_30m, 30)
    print('Original built created at: ', built_file_resized)

    # GHS-POP
    temp_pop_file = os.path.join(temp_clip_path, 'original-spop.tif')
    clip_from_template(pop, reference_file.strip(), temp_pop_file)
    pop_file_resized = os.path.join(output_file_path, '10m-original-spop.tif')
    resize_scene_custom_size(temp_pop_file, pop_file_resized, x_size, x_size)
    pop_file_resized_30m = os.path.join(
        output_file_path, '30m-original-spop.tif')
    change_pixel_size(pop_file_resized, pop_file_resized_30m, 30)
    print('Original spop created at: ', pop_file_resized)

    # GHS-SMOD
    temp_smod_file = os.path.join(temp_clip_path, 'original-smod.tif')
    clip_from_template(smod, reference_file.strip(), temp_smod_file)
    smod_file_resized = os.path.join(output_file_path, '10m-original-smod.tif')
    resize_scene_custom_size(temp_smod_file, smod_file_resized, x_size, x_size)
    smod_file_resized_30m = os.path.join(
        output_file_path, '30m-original-smod.tif')
    change_pixel_size(smod_file_resized, smod_file_resized_30m, 30)
    print('Original smod created at: ', smod_file_resized)



if __name__ == '__main__':
    main()
