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


def merge(input1, bounds, res, nodata, precision):
    import warnings
    warnings.warn("Deprecated; Use rasterio.merge instead", DeprecationWarning)
    return rasterio.merge.merge(input1, bounds, res, nodata, precision)


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


def download_scene(file_id, out_path):
    api = SentinelAPI('<uname>', '<pwd>',
                      api_url='https://scihub.copernicus.eu/apihub')
    return api.download(file_id, directory_path=out_path)


def unzip_scene(file_name, out_dir):
    subprocess.call('unzip '+file_name+' -d '+out_dir, shell=True)


def jp2_to_tif(input_file, output_file):
    subprocess.call('gdal_translate ' + input_file+' '+output_file, shell=True)


def create_new_tif(sourceraster, targetraster, array2d, dtype, nbands):
    with rasterio.open(sourceraster) as src:
        metadata = src.profile
    metadata['count'] = nbands
    metadata['dtype'] = dtype
    with rasterio.open(targetraster, 'w', **metadata) as dst:
        dst.write(array2d, 1)
        print('New tif created at: ', str(targetraster))


def resize_scene_custom_size(input_file, output_file, xsize, ysize):
    subprocess.call('gdal_translate -co QUALITY=100 -co PROGRESSIVE=ON -outsize ' +
                    str(xsize)+' '+str(ysize)+' '+input_file+' '+output_file, shell=True)


def change_pixel_size(inputfile, outputfile, pixelsize):
    subprocess.call('gdalwarp -tr ' + str(pixelsize)+' ' +
                    str(pixelsize)+' '+inputfile+' '+outputfile, shell=True)


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


def preprocess_scene(train_path, scene_folder_name, s2_file_id, reference_template):

    metadata = download_scene(
        s2_file_id, os.path.join(train_path, scene_folder_name))
    unzip_scene(metadata['path'], os.path.join(train_path, scene_folder_name))
    extracted_folder_path = os.path.join(os.path.join(
        train_path, scene_folder_name), metadata['title']+'.SAFE')

    for root, dir, filelist in os.walk(extracted_folder_path):
        if 'IMG_DATA' in root:
            img_data_path = root

    for i in os.listdir(img_data_path):
        print('Processing...'+i)
        jp2_to_tif(os.path.join(img_data_path, i), os.path.join(
            img_data_path, os.path.splitext(i)[0]+'.tif'))

    tif_files_temp = []
    for i in os.listdir(img_data_path):
        if '.tif' in i and '_B' in i:
            tif_files_temp.append(os.path.join(img_data_path, i))
    tif_files_temp.sort()
    tif_files = band_sorter_S2(tif_files_temp)
    print('Sorted list: ', tif_files)
    stack_bands(tif_files, os.path.join(os.path.join(
        train_path, scene_folder_name), scene_folder_name+'-merged.tif'))
    change_pixel_size(os.path.join(os.path.join(train_path, scene_folder_name), scene_folder_name +
                                   '-merged.tif'), os.path.join(os.path.join(train_path, 'X'), scene_folder_name+'.tif'), 30)

    clip_from_vrt(os.path.join(os.path.join(train_path, 'X'), scene_folder_name+'.tif'), reference_template, os.path.join(os.path.join(
        train_path, scene_folder_name), scene_folder_name+'-temp.tif'), os.path.join(os.path.join(train_path, 'Y'), scene_folder_name+'.tif'))


# MAKE DIRECTORIES
os.mkdir('train_data')
os.mkdir('train_data/X')
os.mkdir('train_data/Y')
os.mkdir('train_data/city1')
os.mkdir('train_data/city2')
os.mkdir('train_data/city3')
os.mkdir('train_data/city4')
os.mkdir('train_data/city5')
os.mkdir('train_data/city6')
os.mkdir('train_data/city7')
os.mkdir('train_data/city8')
os.mkdir('train_data/city9')
os.mkdir('train_data/city10')
os.mkdir('train_data/city11')
os.mkdir('train_data/city12')
os.mkdir('train_data/city13')
os.mkdir('train_data/city14')
os.mkdir('train_data/city15')
os.mkdir('reference_data')

# DOWNLOAD REFERENCE DATA
subprocess.call('wget '+'https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_LDSMT_GLOBE_R2018A/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.zip'+' -P '+'reference_data', shell=True)
unzip_scene(
    'reference_data/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.zip', 'reference_data')

print('Downloading reference data done')


# DOWNLOAD S2 IMAGES
preprocess_scene('train_data', 'city1', '4f07815d-5709-4b0b-8fdd-7e12c78e6546',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')

preprocess_scene('train_data', 'city2', '03cf3314-1b05-46f1-af1b-cee2652edf2b',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')

preprocess_scene('train_data', 'city3', '67a87302-2ff4-4c81-9e5e-792367578f6d',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')

preprocess_scene('train_data', 'city4', '4557d038-7f7e-4356-b410-5c6e7d741e7f',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')

preprocess_scene('train_data', 'city5', 'e27091e1-9c59-4326-8b3f-2dfd81d99c65',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')

preprocess_scene('train_data', 'city6', '156b08ab-182b-4867-87a7-4e0696c4b3f4',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')

preprocess_scene('train_data', 'city7', 'ca614e0e-5052-4d4f-8f96-416b39bea39e',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')

preprocess_scene('train_data', 'city8', '8736dc83-d886-4cd8-9ef1-362462228e30',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')

preprocess_scene('train_data', 'city9', '0713bb58-4c01-479f-b126-e75c9d8924b2',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')


preprocess_scene('train_data', 'city10', '6fc05010-23c4-4bcc-87d6-963e44e7f83d',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')


preprocess_scene('train_data', 'city11', '81de1cc5-a9ef-42e5-9125-2d067d9b3475',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')

preprocess_scene('train_data', 'city12', '45a37df2-9c94-4ad6-941e-bf1041e831d0',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')


preprocess_scene('train_data', 'city13', '59615470-4e44-402c-a592-826fba0b964a',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')


preprocess_scene('train_data', 'city14', '45824f90-7962-4a4f-848b-51cd5339b46e',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')


preprocess_scene('train_data', 'city15', '1edcaec7-0837-4ca9-9d43-f92e3064f3fb',
                 'reference_data/V2-0/GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0.vrt')


print('Downloading s2 data done')
