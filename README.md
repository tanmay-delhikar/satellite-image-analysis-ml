# SATELLITE IMAGE ANALYSIS WITH MACHINE LEARNING
### PRESENTATION SLIDES: [project_report (pdf)](/project_report.pdf)

#### **OUTPUT FILES FORMAT**

Folders with results for CNN, SML algorithms are generated within each of the .SAFE output folder. 

| Format  | Example | Use |
| ------ | ------ | ------ |
| {resolution}.tif | 10m.tif | All ordered bands of S2/L8 stacked in a single file with 10m resolution (multispectral) |
| {resolution}-{bands}-{product}-{type}.tif | 10m-9-built-prediction.tif | BUILT prediction(or confidence) file using 9 bands with 10m resolution|
| {resolution}-{bands}-{product}.tif | 10m-9-pop.tif | SML based POP file using 9 bands|
| {resolution}-original-{product}.tif | 10m-original-built.tif | Original GHSL products(BUILT,SPOP,SMOD) clipped as the same region and resolution as input file|
| metrics.json | metrics.json | Metrics calculated by comparing BUILT and Original GHSL-BUILT (meanIOU, F1 score etc.) |



#### **SAMPLE RESULTS**

| Input Data | Results | Algorithm
| ------ | ------ | ------ |
| /netscratch/delhikar/test_data/ | /netscratch/delhikar/results_all_new/ | CNN/SML |
| /netscratch/delhikar/WSF/version2/test_data/ | /netscratch/delhikar/WSF/version2/test_data_results_new/ | WSF |



#### **SYMBOLIC MACHINE LEARNING (SML) BASED ALL PRODUCTS (Multiprocessing mode)**
Generates all SML based products (BUILT,SPOP,SMOD) for a given directory containing a list of Sentinel2(.SAFE) or Landsat8 satellite folders.

| Parameter | Use |
| ------ | ------ |
| -c | Configuration file path |
| -rf | Parent folder containing a list of s2(.SAFE) or l8 folders |
| -of | Output folder |
| -nb | No. of bands to be used for inference |
| -st | Satellite type - s2 or l8 |

Command:
```sh
$ python3 sml-all-products.py -c '<path to config.json>' -rf '<parent folder of bulk s2/l8 images>' -of '<output folder to store results>' -nb <no. of bands> -st '<satellite type- s2 or l9>'
```
Example:
```sh
$ python3 sml-all-products.py -c "/netscratch/delhikar/GHS-SML/version4/config.json" -rf  "/netscratch/delhikar/test_data/" -of  "/netscratch/delhikar/test_data_results" -st "s2"  -nb  9
```

#### **CNN BASED PRODUCTS**
Generates all CNN based products (BUILT,POP,SMOD) for a given root directory containing a list of processed Sentinel2(.SAFE) sub-folders.

| Parameter | Use |
| ------ | ------ |
| -c | Configuration file path |
| -rf | Parent folder containing a list of .SAFE folders |
| -of | Output folder |
| -st | Satellite type - s2 or l8 |

Command:
```sh
$ python3 cnn-all-products.py -c '<path to config.json>' -rf '<parent folder with list of .SAFE folders>' -of '<output folder to store results>' -st '<satellite type- s2 or l9>'
```
Example:
```sh
$ python3 cnn-all-products.py -c "/netscratch/delhikar/GHS-SML/version4/config.json" -rf  "/netscratch/delhikar/test_data/" -of  "/netscratch/delhikar/test_data_results" -st "s2"  -st 's2'
```


#### **SYMBOLIC MACHINE LEARNING (SML) TRAINING DATA DOWNLOADER**

Execute in the current directory you want to download and organize training data for SML automatically. (Change username and password in the code).

Command:
```sh
$ python3 s2-data-downloader.py 
```

#### **SYMBOLIC MACHINE LEARNING (SML) BASED BUILT TRAINING**

Trains SML algorithm using above training data generated in previous step for a given number of bands.

| Parameter | Use |
| ------ | ------ |
| -c | Configuration file path |
| -nb | No. of bands to be used for training |



Command:
```sh
$ python3 sml-built-train.py -c '<path to config.json>' -nb <NO. OF BANDS> 
```
Example:
```sh
$ python3 sml-built-train.py -c "/netscratch/delhikar/GHS-SML/version4/config.json" -nb 9 
```


#### **SYMBOLIC MACHINE LEARNING (SML) BASED BUILT**

Generates SML-BUILT(prediction and confidence) products for a given multispectral image using specified number of bands

| Parameter | Use |
| ------ | ------ |
| -c | Configuration file path |
| -nb | No. of bands to be used for inference |
| -i | A Multi Spectral (MS) TIF file, with 1-12 bands |
| -of | Output folder of results |


Command:
```sh
$ python3 sml-built-generator.py -c '<path to config.json>' -nb <NO. OF BANDS> -i '<path to Multispectral (MS) image>' -of '<output folder path>'
```
Example:
```sh
$ python3 sml-built-generator.py -c "/netscratch/delhikar/GHS-SML/version4/config.json" -nb 9 -i "/netscratch/delhikar/results_all_new/S2A_MSIL1C_20190424T101031_N0207_R022_T33VXF_20190424T153347.SAFE/10m.tif" -of "/netscratch/delhikar/results_all_new/S2A_MSIL1C_20190424T101031_N0207_R022_T33VXF_20190424T153347.SAFE/SML/"
```




#### **POPULATION GRID (POP)**

Generates POP product for a given BUILT(prediction and confidence) products resulting from the above step

| Parameter | Use |
| ------ | ------ |
| -c | Configuration file path |
| -cf | A GHS-BUILT (Confidence) TIF file, with 0-100% |
| -pr | A GHS-BUILT (Prediction) TIF file, with 0,1,2 (0=BUILT UP, 1=OTHERS , 2=WATER) |
| -of | Output folder to store results|



Command:
```sh
$ python3 sml-spop-gpw-count-generator.py -c '<path to config.json>' -cf <'path to GHS-BUILT (confidence) file generated from previous step'> -pr '<path to GHS-BUILT (prediction) file generated from previous step>' -of '<Output folder>'  
```
Example:
```sh
$ python3 sml-spop-gpw-count-generator.py -c "/netscratch/delhikar/GHS-SML/version4/config.json" -cf "/netscratch/delhikar/results_all_new/S2A_MSIL1C_20190424T101031_N0207_R022_T33VXF_20190424T153347.SAFE/SML/10m-9-built-confidence.tif" -pr "/netscratch/delhikar/results_all_new/S2A_MSIL1C_20190424T101031_N0207_R022_T33VXF_20190424T153347.SAFE/SML/10m-9-built-prediction.tif" -of "/netscratch/delhikar/results_all_new/S2A_MSIL1C_20190424T101031_N0207_R022_T33VXF_20190424T153347.SAFE/SML/"
```

#### **HUMAN SETTLEMENT MODEL GRID (SMOD)**

Generates SMOD product from a given BUILT (prediction/confidence) and POP products resulted from the above steps.

| Parameter | Use |
| ------ | ------ |
| -cf | A GHS-BUILT (confidence) TIF file, with 0-100% |
| -pr | A GHS-BUILT (prediction) TIF file, with 0,1,2 (0=BUILT UP, 1=UNKNOWN, 2=WATER) |
| -p | A GHS-POP TIF file |
| -of | Output folder |


Command:
```sh
$ python3 sml-smod-generator.py -cf '<path to confidence file>' -pr 'path to prediction generated from previous step' -p '<path to GHS-POP generated from previous step>' -of '<Output folder>'
```
Example:
```sh
$ python3 sml-smod-generator.py -cf '/netscratch/delhikar/GHS-SML/version3/italy-confidence.tif' -pr '/netscratch/delhikar/GHS-SML/version3/italy-prediction.tif' -p '/netscratch/delhikar/GHS-SML/version3/test/italy-pop.tif' -of '/netscratch/delhikar/GHS-SML/version3/30m/results'
```
#### **WORLD SETTLEMENT FOOTPRINT (WS)-BUILT**

Generates WSF based predictions for a given folder containing multitemporal scenes

| Parameter | Use |
| ------ | ------ |
| -c | Configuration file path |
| -i | Multi Spectral (MS) TIF file, with 3 temporal scenes |


Command:
```sh
$ python3 wsf.py -c '<path to config.json>' -if <INPUT FOLDER - MULTITEMPORAL PROCESSED SCENES> 
```
#### REFERENCES
Mentioned in project report: [project_report (pdf)](/project_report.pdf)
