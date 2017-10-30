BASE=/home/henrye/projects/E2E/data
python preprocess.py -train_src $BASE/trainset-source.tok.feat  -train_tgt $BASE/trainset-target.tok -valid_src $BASE/devset-source.tok.feat -valid_tgt $BASE/devset-target.tok -save_data $BASE/fixed -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab
python train.py -data $BASE/fixed -save_model /tmp/generate_with_features -rnn_size 600 -word_vec_size 600 -batch_size 256 -epochs 20 -report_every 50 -gpuid 1 -encoder_type brnn -copy_attn -dropout 0.3 -global_attention dot -truncated_decoder 100 -share_embeddings
python translate.py -model [model]  -src $BASE/devset-source.tok.unique.feat -gpu 0 -beam_size 30 -batch_size 1 -max_sent_length 500 -output /tmp/gen -n_best 1 -verbose
