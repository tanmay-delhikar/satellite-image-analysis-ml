import argparse
import os
import subprocess

import rasterio


def jp2_to_tif(input_file, output_file):
    subprocess.call('gdal_translate -tr 10 10 ' +
                    input_file+' '+output_file, shell=True)


def resize_scene_custom_size(input_file, output_file, xsize, ysize):
    subprocess.call('gdal_translate -co QUALITY=100 -co PROGRESSIVE=ON -outsize ' +
                    str(xsize)+' '+str(ysize)+' '+input_file+' '+output_file, shell=True)


def stack_bands(file_list, output_file):
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    meta.update(count=len(file_list))

    with rasterio.open(output_file, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


def band_sorter_l8(list_temp):
    sorted_list = []
    for element in list_temp:
        if element.endswith('_B1.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B2.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B3.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B4.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B5.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B6.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B7.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B8.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B9.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B10.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_B11.tif'):
            sorted_list.append(element)
    for element in list_temp:
        if element.endswith('_BQA.tif'):
            sorted_list.append(element)
    return sorted_list


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


def change_pixel_size(inputfile, outputfile, pixelsize):
    subprocess.call('gdalwarp -tr ' + str(pixelsize)+' ' +
                    str(pixelsize)+' '+inputfile+' '+outputfile, shell=True)


def merge_all_tiles(input_safe_path, output_safe_path):
    all_tiles_path = os.path.join(input_safe_path, 't')
    output_tiles_path = os.path.join(output_safe_path, 't')
    if not os.path.exists(output_tiles_path):
        os.makedirs(output_tiles_path)
    subprocess.call("find "+all_tiles_path+" -maxdepth 1 -type f -name '*.tif' > " +
                    os.path.join(output_tiles_path, 'fileList.txt'), shell=True)
    subprocess.call("gdalbuildvrt -input_file_list " + os.path.join(output_tiles_path,
                                                                    'fileList.txt') + " " + os.path.join(output_tiles_path, '10m-original.vrt'), shell=True)
    subprocess.call("gdal_translate -of GTiff -co NUM_THREADS=8 "+os.path.join(output_tiles_path,
                                                                               '10m-original.vrt')+" "+os.path.join(output_tiles_path, '10m-original.tif'), shell=True)

    with rasterio.open(os.path.join(output_tiles_path, '10m-original.tif')) as src:
        prof = src.profile
        x_size = prof['width']
        y_size = prof['height']
    resize_scene_custom_size(os.path.join(output_tiles_path, '10m-original.tif'),
                             os.path.join(output_safe_path, '10m.tif'), y_size, y_size)


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

    tif_files_temp = []
    for i in os.listdir(output_safe_path):
        if '.tif' in i and '_B' in i:
            tif_files_temp.append(os.path.join(output_safe_path, i))
    tif_files_temp.sort()
    tif_files = band_sorter_S2(tif_files_temp)
    stack_bands(tif_files, os.path.join(output_safe_path, '10m.tif'))

    print('Processing done')


def preprocess_l8(input_safe_path, output_safe_path):
    if not os.path.exists(output_safe_path):
        os.makedirs(output_safe_path)

    for i in os.listdir(input_safe_path):
        if i.endswith('.TIF') and '_B' in i:
            print('Processing...'+i)
            with rasterio.open(os.path.join(input_safe_path, i)) as src:
                metadata = src.profile
            resize_scene_custom_size(os.path.join(input_safe_path, i), os.path.join(
                output_safe_path, 'transformed-'+os.path.splitext(i)[0]+'.tif'), metadata['width'], metadata['width'])

    tif_files_temp = []
    for i in os.listdir(output_safe_path):
        print(i)
        if '.tif' in i and '_B' in i and 'transformed' in i:
            tif_files_temp.append(os.path.join(output_safe_path, i))
    tif_files_temp.sort()
    tif_files = band_sorter_l8(tif_files_temp)

    stack_bands(tif_files, os.path.join(output_safe_path, '10m.tif'))

    print('Processing done')


def main():
    parser = argparse.ArgumentParser(description="Preprocess S2/L8 folder")
    parser.add_argument("-ip", "--inputPath",
                        help="Root folder containing list of .SAFE folders")
    parser.add_argument("-op", "--outputPath",
                        help="Output path to save preprocessed files")
    parser.add_argument("-st", "--satelliteType",
                        help="Satellite type s2 OR l8")

    args = parser.parse_args()
    input_path = args.inputPath
    output_path = args.outputPath
    st = args.satelliteType

    if(input_path is None):
        print("Root folder containing .SAFE directories is missing")
        exit()

    if(output_path is None):
        print("Output path is missing")
        exit()

    if(st is None):
        print("Satellite type is missing")
        exit()

    if 's2' in st.lower():
        preprocess_s2(input_path, output_path)

    elif 'l8' in st.lower():
        preprocess_l8(input_path, output_path)

    print('Done')


if __name__ == '__main__':
    main()
