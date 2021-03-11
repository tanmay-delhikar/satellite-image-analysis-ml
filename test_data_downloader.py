
import subprocess

from sentinelsat.sentinel import SentinelAPI


def unzip_scene(file_name):
    subprocess.call('unzip '+file_name, shell=True)


def download_scene(file_id):
    api = SentinelAPI('<uname>', '<pwd>',
                      api_url='https://scihub.copernicus.eu/apihub')
    return api.download(file_id)


metadata = download_scene('742eb824-df10-4402-866e-0d92e8575aa4')
unzip_scene(metadata['path'])

metadata = download_scene('03cf3314-1b05-46f1-af1b-cee2652edf2b')
unzip_scene(metadata['path'])

metadata = download_scene('67a87302-2ff4-4c81-9e5e-792367578f6d')
unzip_scene(metadata['path'])

metadata = download_scene('4557d038-7f7e-4356-b410-5c6e7d741e7f')
unzip_scene(metadata['path'])

metadata = download_scene('e27091e1-9c59-4326-8b3f-2dfd81d99c65')
unzip_scene(metadata['path'])

metadata = download_scene('ca614e0e-5052-4d4f-8f96-416b39bea39e')
unzip_scene(metadata['path'])

metadata = download_scene('8736dc83-d886-4cd8-9ef1-362462228e30')
unzip_scene(metadata['path'])

metadata = download_scene('0713bb58-4c01-479f-b126-e75c9d8924b2')
unzip_scene(metadata['path'])

metadata = download_scene('6fc05010-23c4-4bcc-87d6-963e44e7f83d')
unzip_scene(metadata['path'])

metadata = download_scene('81de1cc5-a9ef-42e5-9125-2d067d9b3475')
unzip_scene(metadata['path'])

metadata = download_scene('45a37df2-9c94-4ad6-941e-bf1041e831d0')
unzip_scene(metadata['path'])

metadata = download_scene('59615470-4e44-402c-a592-826fba0b964a')
unzip_scene(metadata['path'])

metadata = download_scene('45824f90-7962-4a4f-848b-51cd5339b46e')
unzip_scene(metadata['path'])

metadata = download_scene('1edcaec7-0837-4ca9-9d43-f92e3064f3fb')
unzip_scene(metadata['path'])
