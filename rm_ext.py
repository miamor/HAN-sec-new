import os
import shutil

for lbl in os.listdir('/media/tunguyen/TuTu_Passport/MTAAV_data/data_report/TuTu__'):
    for filename in os.listdir('/media/tunguyen/TuTu_Passport/MTAAV_data/data_report/TuTu__/'+lbl):
        filename_no_et = filename.split('.')[0]
        shutil.move()