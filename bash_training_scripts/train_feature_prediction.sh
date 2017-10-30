BASE=/home/henrye/projects/E2E/data
python preprocess.py -train_src $BASE/trainset-additional-words-source.tok.filter  -train_tgt $BASE/trainset-additional-words-target.tok.filter -valid_src $BASE/devset-additional-words-source.tok.filter -valid_tgt $BASE/devset-additional-words-target.tok.filter  -save_data $BASE/features_model -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab
python train.py -data $BASE/features_model -save_model /tmp/features_model -rnn_size 600 -word_vec_size 600 -batch_size 32 -epochs 20 -report_every 50 -gpuid 1 -encoder_type mean -dropout 0.3 -global_attention dot -truncated_decoder 100
python translate.py -model features_model*e20.pt -src $BASE/devset-additional-words-source.tok.filter -gpu 0 -beam_size 30 -batch_size 1 -max_sent_length 500 -output /tmp/gen_features -n_best 1 -verbose

