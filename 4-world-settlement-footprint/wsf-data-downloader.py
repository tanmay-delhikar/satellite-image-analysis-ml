import argparse
import codecs
import json
import multiprocessing
import os
import pickle
import subprocess

from sentinelsat.sentinel import SentinelAPI
from tqdm import tqdm

import gdal
import numpy as np
import rasterio
from gdalconst import GA_ReadOnly
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals.joblib import parallel_backend
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def download_scene(file_id, out_path):
    api = SentinelAPI('<uname>', '<pwd>',
                      api_url='https://scihub.copernicus.eu/apihub')
    return api.download(file_id, directory_path=out_path)


def unzip_scene(file_name, out_dir):
    subprocess.call('unzip '+file_name+' -d '+out_dir, shell=True)


def check_or_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def jp2_to_tif(input_file, output_file):
    subprocess.call('gdal_translate -tr 10 10 ' +
                    input_file+' '+output_file, shell=True)


# DOWNLOAD TRAINING DATA
scene_ids = ['8a2f80f8-0db0-4cf3-9dba-607aa2587329',
             'ca614e0e-5052-4d4f-8f96-416b39bea39e', '77964312-dc57-4e0b-835a-3a6dac71ac6c']


for i in scene_ids:
    metadata = download_scene(i, '.')
    unzip_scene(metadata['path'], '.')
