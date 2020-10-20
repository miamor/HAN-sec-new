import os
import shutil
import random

trains = []
tests = []


SET = 'TuTu'


ROOT = '../../MTAAV_data/data_report/'+SET+'/'

for lbl in os.listdir(ROOT):
    tot_files = len([name for name in os.listdir(ROOT+lbl) if os.path.isfile(os.path.join(ROOT+lbl, name))])

    train_end_idx = tot_files*0.75
    print(lbl, 'train_end_idx', train_end_idx)

    n = 0
    for filename in os.listdir(ROOT+lbl):
        n += 1

        if n <= train_end_idx:
            # train
            trains.append('{}__{}'.format(lbl, filename))
        else:
            tests.append('{}__{}'.format(lbl, filename))
    print('n', n)
print('trains', len(trains))
print('tests', len(tests))

'''
trains = []
tests = []
with open('/media/fit/TuTu_Passport/MTAAV/HAN-sec-new/data_pickle/reverse/TuTu__vocabtutu__iapi__tfidf__topk=3/graphs_name.txt', 'r') as f:
    n = {'benign':0, 'malware': 0}
    tot_files = {'benign': 1789, 'malware': 1875}
    train_end_idx = {
        'benign': tot_files['benign']*0.75,
        'malware': tot_files['malware']*0.75
    }
    print('train_end_idx', train_end_idx)

    lines = [line.strip() for line in f.readlines()]
    
    random_indices = list(range(len(lines)))
    random.shuffle(random_indices)
    lines = [lines[i] for i in random_indices]

    for line in lines:
        lbl = line.split('__')[0]
        filename = line.split('{}__'.format(lbl))[1]
        n[lbl] += 1
        
        if n[lbl] <= train_end_idx[lbl]:
            # train
            trains.append('{}__{}'.format(lbl, filename))
        else:
            tests.append('{}__{}'.format(lbl, filename))
'''

# ROOT = 'data_graphs/reverse/TuTu__vocabtutu__iapi__tfidf__topk=3___/'

# for lbl in os.listdir(ROOT):
#     if not os.path.isdir(ROOT+lbl):
#         continue

#     tot_files = len([name for name in os.listdir(ROOT+lbl) if os.path.isfile(os.path.join(ROOT+lbl, name))])

#     train_end_idx = tot_files*0.75
#     print(lbl, 'train_end_idx', train_end_idx)

#     n = 0
#     for filename in os.listdir(ROOT+lbl):
#         n += 1

#         if n <= train_end_idx:
#             # train
#             trains.append(filename+'.json')
#         else:
#             tests.append(filename+'.json')

#         if os.path.exists('../../MTAAV_data/data_report/TuTu__/'+lbl+'/'+filename.split(lbl+'__')[1]+'.json'):
#             shutil.move('../../MTAAV_data/data_report/TuTu__/'+lbl+'/'+filename.split(lbl+'__')[1]+'.json', '../../MTAAV_data/data_report/TuTu/'+lbl+'/'+filename.split(lbl+'__')[1]+'.json')
#         if os.path.exists('../../MTAAV_data/data_report/TuTu__/'+lbl+'/'+filename.split(lbl+'__')[1]+'.dll.json'):
#             shutil.move('../../MTAAV_data/data_report/TuTu__/'+lbl+'/'+filename.split(lbl+'__')[1]+'.dll.json', '../../MTAAV_data/data_report/TuTu/'+lbl+'/'+filename.split(lbl+'__')[1]+'.dll.json')
#         if os.path.exists('../../MTAAV_data/data_report/TuTu__/'+lbl+'/'+filename.split(lbl+'__')[1]+'.exe.json'):
#             shutil.move('../../MTAAV_data/data_report/TuTu__/'+lbl+'/'+filename.split(lbl+'__')[1]+'.exe.json', '../../MTAAV_data/data_report/TuTu/'+lbl+'/'+filename.split(lbl+'__')[1]+'.exe.json')


with open('data/'+SET+'_train_list.txt', 'w') as f:
    f.write('\n'.join(trains))

with open('data/'+SET+'_test_list.txt', 'w') as f:
    f.write('\n'.join(tests))
