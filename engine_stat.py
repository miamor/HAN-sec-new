import os
import json

dir = '/media/fitmta/Storage/MinhTu/Dataset/cuckoo_ADung_none/malware'

for filename in os.listdir(dir):
    filepath = dir+'/'+filename
    
    with open(filepath) as json_file:
        data = json.load(json_file)
        
        info = data['info']