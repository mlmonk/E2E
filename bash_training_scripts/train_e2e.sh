export CUDA_VISIBLE_DEVICES=1
BASE=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/data
MODEL=baseline_test_gpu_0
MODEL_DIR=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/model
TGEN=$BASE/tgen
OUTPUT=$BASE/output
GPU=0
TOTAL_GPUS=1
python preprocess.py -train_src $BASE/trainset-source.tok  -train_tgt $BASE/trainset-target.tok -valid_src $BASE/devset-source.tok -valid_tgt $BASE/devset-target.tok -save_data $BASE/$MODEL -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab
python train.py -data $BASE/$MODEL -save_model $MODEL_DIR/$MODEL -rnn_size 60 -word_vec_size 600 -batch_size 256 -train_steps 20 -report_every 50 -world_size=$TOTAL_GPUS -gpu_ranks $GPU -encoder_type brnn -copy_attn -dropout 0.3 -global_attention dot -truncated_decoder 100

for model_checkpoint in $MODEL_DIR/$MODEL{10..20}.pt
    do 
        echo $model_checkpoint
        python translate.py -model $model_checkpoint -src $BASE/devset-source.tok -gpu $GPU -beam_size 30 -batch_size 1 -max_length 500 -output $OUTPUT -n_best 1
        cd /home/nikhil/Projects/e2e-metrics
	source activate Python2
        # measure scores run on python 2 so we switch back to the theano environment
        python measure_scores.py $TGEN/devset-target.grouped $OUTPUT
	source deactivate
        cd /home/nikhil/OpenNMT-py
    done

