model_checkpoint = ''
beam_size = None
temperate = None
devset_source = ''
testset_source = ''
gpu = None
output_folder = ''


base=/home/henrye/projects/E2E/data
model=baseline_test_gpu_0
gpu=0



python preprocess.py -train_src $base/trainset-source.tok  -train_tgt $base/trainset-target.tok -valid_src $base/devset-source.tok -valid_tgt $base/devset-target.tok -save_data $base/$model -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab
python train.py -data $base/$model -save_model /tmp/$model -rnn_size 600 -word_vec_size 600 -batch_size 256 -epochs 20 -report_every 50 -gpuid $gpu -encoder_type brnn -copy_attn -dropout 0.3 -global_attention dot -truncated_decoder 100


for model_checkpoint in /tmp/$model*{10..20}.pt
    do 
        echo $model_checkpoint
        python translate.py -model $model_checkpoint -src $base/devset-source.tok.unique -gpu $gpu -beam_size 30 -batch_size 1 -max_sent_length 500 -output /tmp/gen -n_best 1
        cd /home/henrye/downloads/e2e-metrics
        # measure scores run on python 2 so we switch back to the theano environment
        source activate theano
        python measure_scores.py ~/projects/E2E/data/devset-target.grouped /tmp/gen
        cd /home/henrye/downloads/OpenNMT-py
        source activate pytorch
    done