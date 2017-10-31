data_dir=/home/henrye/projects/E2E/data
experiment_dir=/home/henrye/projects/E2E/experiments
model_type='additional_words_generator'
now=`date '+%Y_%m_%d__%H_%M'`
model_dir="$experiment_dir/$model_type-$now"
mkdir $model_dir

source activate pytorch
current_dir=`pwd`
cd ~/downloads/OpenNMT-py
python preprocess.py \
    -train_src $data_dir/trainset-additional-words-source.tok.filter \
    -train_tgt $data_dir/trainset-additional-words-target.tok.filter \
    -valid_src $data_dir/devset-additional-words-source.tok.filter \
    -valid_tgt $data_dir/devset-additional-words-target.tok.filter \
    -save_data $model_dir/preprocessed \
    -src_seq_length 1000 -tgt_seq_length 1000 
    # -dynamic_dict \
    # -share_vocab
    cp $data_dir/{devset-additional-words-source.tok.unique,test_e2e-additional-words-source.tok} $model_dir/
cd $current_dir

