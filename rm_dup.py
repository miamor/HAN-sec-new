import os

l = []
with open('/media/fitmta/Storage/MinhTu/HAN_sec_good/data_vocab/vocablower_noiapi_full/node.txt', 'r+') as f:
    x = f.readline().strip().split(' ')
    for w in x:
        if w not in l:
            l.append(w)
    
    print('l', l)

    f.seek(0)
    f.truncate()
    f.write(' '.join(l))
