# -*- coding: utf-8 -*-

'''
This file is used to process DataSet DAIC-WOZ, including two steps:
1. copy .csv files to target folder.
2. unzip zip files to target folder.
you can change file path in main function, the log file name is preprocessing.log.
'''
import os
import glob
import zipfile
import shutil
import logging

def copy_csv_files(COPY_SOURCE_PATH,COPY_TARGET_PATH):
    for CSVFile in glob.iglob(os.path.join(COPY_SOURCE_PATH,"*.csv")):
        shutil.copy(CSVFile, COPY_TARGET_PATH)
        logging.info("Copied file " + CSVFile + " to " + COPY_TARGET_PATH)

def unzip_file(zip_src, dst_dir):
    logging.info("Will extract file: " + zip_src +" to " + dst_dir)
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
        logging.info("Success unzip file: " + zip_src)  
    else:
        logging.info('This is not zip')
 
def load_unzip(ZIP_PATH, EXTRACT_BASE_PATH):
    for i in range(300,493):
        zip_file_name = ZIP_PATH + "/" + str(i) + "_P.zip"
        extracted_file_name = EXTRACT_BASE_PATH + "/" + str(i) + "_P"
        unzip_file(zip_file_name,extracted_file_name)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=r'preprocessing\preprocessing.log',
                    filemode='w')
    
    SOURCE_PATH=r'/your_data_path/DAIC-WOZ'
    TARGET_PATH=r'/your_data_path/DAIC-WOZ-Unzip'
    copy_csv_files(SOURCE_PATH,TARGET_PATH)

    load_unzip(SOURCE_PATH,TARGET_PATH)


