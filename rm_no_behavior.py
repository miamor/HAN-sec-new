import os
import json
import shutil

ROOT = '../../MTAAV_data/data_report/TuTuTest'

for lbl in os.listdir(ROOT):
    n = 0
    for filename in os.listdir(ROOT+'/'+lbl):
        n += 1
        print(n, lbl+'/'+filename)
        content = json.load(open(ROOT+'/'+lbl+'/'+filename, 'r'))
        if 'behavior' not in content or len(content['behavior']['processes']) == 0:
            shutil.move(ROOT+'/'+lbl+'/'+filename, ROOT+'__no_behavior/'+lbl+'/'+filename)
        