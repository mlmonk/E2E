
# We're just trying out a bunch of different things


for training_opt in '\"-word_vec_size 1024 -rnn_size 600 -layers 2 -dropout 0.3\"' '\"-word_vec_size 1024 -rnn_size 1024 -layers 2 -dropout 0.5\"'
do
    bash ~/projects/E2E/bash_training_scripts/start_training_additional_words_generator.sh \
        -n boole-n023 \
        -g 1 \
        -p /home/henrye/projects/E2E/experiments/additional_words_generator-2017_10_31__09_32 \
        -t $training_opt
done


# now to do the same thing for evaluation 
for checkpoint_dir in /home/henrye/projects/E2E/experiments/additional_words_generator-2017_10_31__09_32/checkpoints-boole-n023-gpu_1-2017_10_31__10_02 /home/henrye/projects/E2E/experiments/additional_words_generator-2017_10_31__09_32/checkpoints-boole-n023-gpu_1-2017_10_31__09_51
do
    for temperature in 0.4 0.5 0.6 0.7 0.8
    do
        for beam_size in 1
        do
            evaluation_opts="\"-temperature $temperature -beam_size $beam_size\""
            echo $evaluation_opts

            bash ~/projects/E2E/bash_training_scripts/evaluate_additional_words_checkpoint.sh  \
                -n boole-n023 \
                -g 0 \
                -c $checkpoint_dir \
                -e "$evaluation_opts"
        done
    done
done

# temperature=0.5
# beam_size=5
# checkpoint_dir=/home/henrye/projects/E2E/experiments/additional_words_generator-2017_10_31__09_32/checkpoints-boole-n023-gpu_1-2017_10_31__10_02
# evaluation_opts="\"-temperature $temperature -beam_size $beam_size\""
# echo $evaluation_opts
# echo $($evaluation_opts)
# bash ~/projects/E2E/bash_training_scripts/evaluate_additional_words_checkpoint.sh  \
#     -n boole-n023 \
#     -g 0 \
#     -c $checkpoint_dir \
#     -e "$evaluation_opts"

