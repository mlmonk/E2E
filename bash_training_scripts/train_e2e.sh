!/bin/bash
export CUDA_VISIBLE_DEVICES=1
BASE=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/data
MODEL=baseline_test_gpu_0
GPU=0
TOTAL_GPUS=1
python preprocess.py -train_src $BASE/trainset-source.tok  -train_tgt $BASE/trainset-target.tok -valid_src $BASE/devset-source.tok -valid_tgt $BASE/devset-target.tok -save_data $BASE/$MODEL -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab
python train.py -data $BASE/$MODEL -save_model /home/nikhil/$MODEL -rnn_size 6 -word_vec_size 600 -batch_size 256 -train_steps 10 -report_every 10 -world_size=$TOTAL_GPUS -gpu_ranks $GPU -encoder_type brnn -copy_attn -dropout 0.3 -global_attention dot -truncated_decoder 100

/home/nikhil/baseline_test_gpu_0_step_10.pt

for model_checkpoint in /home/nikhil/baseline_test_gpu_0_step_10.pt
    do 
        echo $model_checkpoint
        python translate.py -model $model_checkpoint -src $BASE/devset-source.tok.unique -gpu $GPU -beam_size 30 -batch_size 1 -max_length 500 -output /home/nikhil/gen/gen -n_best 1
        cd /home/nikhil/Projects/e2e-metrics
	source activate Python2
        # measure scores run on python 2 so we switch back to the theano environment
        python measure_scores.py $BASE/devset-target.grouped /home/nikhil/gen/gen
	source deactivate
        cd /home/nikhil/OpenNMT-py
    done

