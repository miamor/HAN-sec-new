python3 main.py \
 -r ../Dataset/cuckoo_ADung \
 -d data_json/cuckoo_ADung \
 -p data_pickle/cuckoo_ADung__iapi__vocablower_iapi__doc2vec \
 -v data_vocab/vocablower_iapi \
 -e data_embedding/cuckoo_ADung__iapi__doc2vec \
 -m data/mapping_benign_malware.json \
 -fr -fd -a -pv -tf \
 -train data/train_list_8563.txt \
 -test data/test_list_8563.txt \
train \
 --lr 0.001 --weight_decay 0.001 --batch_size 16 --k_fold 10


python3 main.py \
 -r ../Dataset/cuckoo_ADung \
 -d data_json/cuckoo_ADung \
 -p data_pickle/cuckoo_ADung__iapi__vocablower_iapi__doc2vec \
 -v data_vocab/vocablower_iapi \
 -e data_embedding/cuckoo_ADung__iapi__doc2vec \
 -m data/mapping_benign_malware.json \
 -fp \               
test \
 -o output/ft_type+lbl__9266__666__cuckoo_ADung__iapi__vocablower_iapi__doc2vec


prep_data__6_types_old.py