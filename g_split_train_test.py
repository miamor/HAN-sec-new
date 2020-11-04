import pickle
import torch
from utils.utils import *
import random


SKIP_SYS = False


data_dir_src = 'data_pickle/reverse/TuTu__vocabtutu__iapi__tfidf__topk=10_lv=word'
# data_dir_dst = 'data_pickle/reverse/TuTu_nosys__vocabtutu__iapi__tfidf__topk=10_lv=word'
data_dir_dst = 'data_pickle/reverse/TuTu__vocabtutu__iapi__tfidf__topk=10_lv=word__new'

if not os.path.exists(data_dir_dst):
    os.makedirs(data_dir_dst)


bin_root = '../../MTAAV_data/bin/TuTu/'
rp_root = '../../MTAAV_data/data_report/TuTu/'

map_hash_bin = {}

for lbl in ['benign', 'malware']:
    for file in os.listdir(bin_root+lbl):
        fnr = list(filter(None, file[:-5].split('__')))
        # print(fnr)
        hash = fnr[0]
        map_hash_bin[hash] = file[:-5]
    print('map_hash_bin', map_hash_bin)


# for set in ['train', 'test']:
#     gnames = load_pickle(data_dir+'/graphs_name_'+set)
#     print('gnames', set, gnames)
#     with open(data_dir+'/graphs_name_'+set+'.txt', 'w') as f:
#         f.write('\n'.join(gnames))


print(data_dir_src+'/graphs')
G = load_pickle(data_dir_src+'/graphs')
lbls = torch.load(data_dir_src+'/labels')
with open(data_dir_src+'/graphs_name.txt') as f:
    g_names = [line.strip() for line in f.readlines()]
gnames = load_pickle(data_dir_src+'/graphs_name')

print('g_names', g_names)
print('lbls', len(lbls))
print('g_names', len(g_names))

random_indices = list(range(len(g_names)))
random.shuffle(random_indices)
G = [G[i] for i in random_indices]
lbls = lbls[random_indices]
g_names = [g_names[i] for i in random_indices]
gnames = [gnames[i] for i in random_indices]




n_bgn = (lbls == 0).sum().item()
if SKIP_SYS:
    n_bgn -= 433
n_mal = (lbls == 1).sum().item()
print('bgn', n_bgn)
print('mal', n_mal)

if n_mal > n_bgn+100:
    n_mal = n_bgn + 100

n_tot = {
    0: n_bgn,
    1: n_mal
}
train_idx = {
    0: 0.7*n_bgn,
    1: 0.7*n_mal
}

G_new = {
    'train': [],
    'test': []
}
lbls_new = {
    'train': [],
    'test': []
}
g_names_new = {
    'train': [],
    'test': []
}
gnames_new = {
    'train': [],
    'test': []
}
new_added = {
    0: 0,
    1: 0
}
n = {
    0: 0,
    1: 0
}
for k,lbl in enumerate(lbls):
    # lbl = lbls[k].item()
    lbl = lbl.item()
    g = G[k]
    gname = gnames[k]
    g_name = g_names[k]

    set = None
    if n[lbl] < train_idx[lbl]:
        set = 'train'
    elif n[lbl] <= n_tot[lbl]:
        set = 'test'

    g_name = os.path.splitext(g_name)[0] # remove last .json of report filename

    fnr = list(filter(None, g_name.split('__')))
    
    if SKIP_SYS:
        if lbl == 0 and len(fnr) <= 3:
            print('sys file. SKIP', fnr)
            set = None

    bin_name = '__'.join(fnr[2:])
    if lbl == 1 and len(fnr) <= 3:
        bin_name = bin_name+'__'

    if not os.path.exists('../../MTAAV_data/bin/TuTu/'+fnr[0]+'/'+bin_name):
        setnone = True
        if lbl == 1: # malware, check if is in none set
            hash = fnr[2]
            if hash in map_hash_bin:
                bin_name_new = map_hash_bin[hash]
                g_name = fnr[0]+'__'+fnr[1]+'__'+bin_name_new
                setnone = False
                print('in NONE set.', hash, '~~', map_hash_bin[hash], '~~', g_name)

        if setnone:
            print('bin not found. SKIP', fnr[0]+'/'+bin_name)
            set = None

    if set is not None:
        n[lbl] += 1
        G_new[set].append(g)
        lbls_new[set].append(lbl)
        gnames_new[set].append(gname)
        g_names_new[set].append(g_name)


for lbl in [0,1]:
    print(lbl, n[lbl])


for set in ['train', 'test']:
    print('***', set)
    print('gnames_new', set, len(gnames_new[set]))

    # print('G_new[set]', G_new[set])
    save_pickle(G_new[set], data_dir_dst+'/graphs_'+set)
    save_pickle(gnames_new[set], data_dir_dst+'/graphs_name_'+set)

    lbls_new[set] = torch.tensor(lbls_new[set])
    torch.save(lbls_new[set], data_dir_dst+'/labels_'+set)
    
    with open(data_dir_dst+'/graphs_name_'+set+'.txt', 'w') as f:
        f.write('\n'.join(g_names_new[set]))
