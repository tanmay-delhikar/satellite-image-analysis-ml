import argparse
import multiprocessing
import os
import subprocess


def generate_sml_all_products(safe_folder_path,output_folder_path,satellite_type,nbands,config_json_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
  
    subprocess.call('python3 preprocessor.py -ip '+safe_folder_path+' -op '+output_folder_path+' -st '+satellite_type,shell=True)
    file_name='10m.tif'
    base_name=os.path.splitext(os.path.basename(file_name))[0]+'-'+str(nbands)
    INPUT_FILE=os.path.join(output_folder_path,file_name)
    print(base_name)

    #SML BASED PREDICTIONS
    SML_OUTPUT_PATH=os.path.join(output_folder_path,'SML')
    if not os.path.exists(SML_OUTPUT_PATH):
        os.makedirs(SML_OUTPUT_PATH) 

    BUILT_CONFIDENCE_FILE=os.path.join(SML_OUTPUT_PATH,base_name+'-built-confidence.tif')
    BUILT_PREDICTION_FILE=os.path.join(SML_OUTPUT_PATH,base_name+'-built-prediction.tif')
    POP_FILE=os.path.join(SML_OUTPUT_PATH,base_name+'-pop.tif')
    subprocess.call('python3 sml-built-generator.py -c '+config_json_path+' -nb '+str(nbands)+' -i '+INPUT_FILE+' -of '+SML_OUTPUT_PATH,shell=True)
    subprocess.call('python3 sml-spop-gpw-count-generator.py -c '+config_json_path+' -cf '+BUILT_CONFIDENCE_FILE+' -pr '+BUILT_PREDICTION_FILE+' -of '+SML_OUTPUT_PATH,shell=True)
    subprocess.call('python3 sml-smod-generator.py '+' -cf '+BUILT_CONFIDENCE_FILE+' -pr '+BUILT_PREDICTION_FILE+' -p '+POP_FILE+' -of '+SML_OUTPUT_PATH,shell=True)


    #ORGINAL REFERNCE FILES
    ORIGINAL_OUTPUT_PATH=os.path.join(output_folder_path,'ORIGINAL')
    if not os.path.exists(ORIGINAL_OUTPUT_PATH):
        os.makedirs(ORIGINAL_OUTPUT_PATH) 
    ORIGINAL_BUILT_PREDICTION_FILE=os.path.join(ORIGINAL_OUTPUT_PATH,'10m-original-built.tif')
    subprocess.call('python3 sml-original-products-clipper.py -rf '+INPUT_FILE+' -c '+config_json_path+' -of '+ORIGINAL_OUTPUT_PATH,shell=True)


    #METRICS
    subprocess.call('python3 sml-metrics.py -pr '+BUILT_PREDICTION_FILE+' -og '+ORIGINAL_BUILT_PREDICTION_FILE+' -of '+SML_OUTPUT_PATH,shell=True)
    print('Done generating results for: ',output_folder_path)


def main():
    parser = argparse.ArgumentParser(description="Generate SML based BUILT,POP,SMOD products")
    parser.add_argument("-c", "--configJson", help="Config json path")
    parser.add_argument("-rf", "--rootFolder", help="Root folder containing list of .SAFE folders")
    parser.add_argument("-of", "--outputFolder", help="Output path to save preprocessed files")
    parser.add_argument("-st", "--satelliteType", help="Satellite type s2 OR l8")
    parser.add_argument("-nb", "--nBands", help="Satellite type s2 OR l8")


    args = parser.parse_args()
    config_json_path=args.configJson
    root_foler = args.rootFolder
    output_folder = args.outputFolder
    st=args.satelliteType
    nbands=args.nBands

    if(config_json_path is None or root_foler is None or output_folder is None ):
        print("Either of config json, root folder, output folder is missing")
        exit()

    if( st is None or nbands is None or st is None):
        print("Either of satellite type or number of bands or satellite type is missing")
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    SAFE_FOLDERS_LIST=next(os.walk(root_foler))[1]
    SAFE_FOLDERS_LIST.sort()

    pool = multiprocessing.Pool(5) #use cores

    for safe_path in SAFE_FOLDERS_LIST:
        final_input_path=os.path.join(root_foler,safe_path)
        final_output_path=os.path.join(output_folder,safe_path)
        pool.apply_async(generate_sml_all_products, args=(final_input_path,final_output_path,st,int(nbands),config_json_path))

    pool.close()
    pool.join()
    print('All done!')

    


if __name__ == '__main__':
    main()
