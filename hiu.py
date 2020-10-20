
old = []
new = []
for set in ['train', 'test']:
    with open('/media/tunguyen/TuTu_Passport/MTAAV/HAN-sec-new/output/2020-10-06_07-53-59/TuTu_train_test/'+set+'_list.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            filename = line.split('.json')[0].split('__')[-1]
            lbl = line.split('__')[0]
            if lbl == 'benign':
                print('filename', filename)
                if len(filename) > 0:
                    new.append(line)
                else:
                    old.append(line)

with open('data/old_bgn_n.txt', 'w') as f:
    f.write('\n'.join(old))
with open('data/new_bgn_n.txt', 'w') as f:
    f.write('\n'.join(new))