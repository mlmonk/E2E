export CUDA_VISIBLE_DEVICES=1
BASE=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/data
MODEL=struc_train
MODEL_DIR=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/model
TGEN=$BASE/tgen
OUTPUT=$BASE/output_$MODEL
OPENNMT=/home/nikhil/OpenNMT-py
GPU=0
TOTAL_GPUS=1
python $OPENNMT/preprocess.py -train_src $BASE/trainset-struc-source.tok  -train_tgt $BASE/trainset-struc-target.tok -valid_src $BASE/devset-struc-source.tok -valid_tgt $BASE/devset-struc-target.tok -save_data $BASE/$MODEL -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab
python $OPENNMT/train.py -data $BASE/$MODEL -save_model $MODEL_DIR/$MODEL -rnn_size 500 -word_vec_size 500 -batch_size 64 -train_steps 6600 -report_every 50 -world_size=$TOTAL_GPUS -gpu_ranks $GPU -encoder_type brnn -copy_attn -dropout 0.3 -global_attention dot --start_decay_steps 528
