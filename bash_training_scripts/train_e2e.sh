BASE=/home/henrye/projects/E2E/data
MODEL=baseline_test_gpu_0
GPU=0
python preprocess.py -train_src $BASE/trainset-source.tok  -train_tgt $BASE/trainset-target.tok -valid_src $BASE/devset-source.tok -valid_tgt $BASE/devset-target.tok -save_data $BASE/$MODEL -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab
python train.py -data $BASE/$MODEL -save_model /tmp/$MODEL -rnn_size 600 -word_vec_size 600 -batch_size 256 -epochs 20 -report_every 50 -gpuid $GPU -encoder_type brnn -copy_attn -dropout 0.3 -global_attention dot -truncated_decoder 100
for model_checkpoint in /tmp/$MODEL*{10..20}.pt
    do 
        echo $model_checkpoint
        python translate.py -model $model_checkpoint -src $BASE/devset-source.tok.unique -gpu $GPU -beam_size 30 -batch_size 1 -max_sent_length 500 -output /tmp/gen -n_best 1
        cd /home/henrye/downloads/e2e-metrics
        # measure scores run on python 2 so we switch back to the theano environment
        source activate theano
        python measure_scores.py ~/projects/E2E/data/devset-target.grouped /tmp/gen
        cd /home/henrye/downloads/OpenNMT-py
        source activate pytorch
    done

