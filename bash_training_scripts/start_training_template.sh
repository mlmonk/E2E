#!/bin/bash

# Usage
# bash ~/projects/E2E/bash_training_scripts/start_training_template.sh \
#     -n boole-n021 \
#     -g 0 \
#     -p /home/henrye/projects/E2E/experiments/features-layers,dropout,vec_sizes-27893-2017_10_31__18_30

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -n|--node)
    node="$2"
    shift # past argument
    shift # past value
    ;;
    -g|--gpu)
    gpu="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--preprocessed_dir)
    preprocessed_dir="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--training_opts)
    training_opts="$2"
    shift # past argument
    shift # past value
    ;;
    -e|--evaluation_opts)
    evaluation_opts="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# make checkpoint directory
# SSH into node
# activate pytorch
# start training
# run evaluation after finished training

now=`date '+%Y_%m_%d__%H_%M'`
# TODO inlcude the suggestion id from sigopt in the folder name
checkpoint_dir="$preprocessed_dir/checkpoints-$node-gpu_$gpu-$now" 
if [ ! -d "$checkpoint_dir" ]; then
    mkdir "$checkpoint_dir"
else
    echo "directory already exists, you must be doing testing"
    checkpoint_dir="$checkpoint_dir-test"
    mkdir $checkpoint_dir
fi



# We need to add backslashes before all the variables assigned during ssh
# https://stackoverflow.com/questions/13032409/ssh-remote-variable-assignment
ssh -t -t $node << EOF
    source activate pytorch
    cd ~/downloads/OpenNMT-py

    # test output for tracer program
    # touch $checkpoint_dir/checkpoint_e8.pt

    # for testing we have epochs set to 1
    python train.py \
        -data $preprocessed_dir/preprocessed \
        -save_model $checkpoint_dir/checkpoint \
        -gpuid $gpu \
        -epochs 10 \
        -report_every 50 \
        -truncated_decoder 100 \
        -batch_size 128 \
        -encoder_type brnn \
        -copy_attn \
        -coverage_attn \
        -optim adam \
        -learning_rate 0.001 \
        -global_attention dot \
        \$training_opts
    # because of how pass the arguments, this doesn't seem to work unless
    # the $ is escaped, even though it's defined before the ssh command
    # It has something to do with us using quotes. The whole string is
    # treated as a single argument by the training script
    touch $checkpoint_dir/.training_finished
    echo 'the checkpoint directory is $checkpoint_dir end transmission'
    exit
EOF

# the evaluation script is called here because we can't think of a better way
# to pass the information about the checkpoint directory. Otherwise we 
# should call evaluation separately. Actually the way we understand it now
# we could probably pass the checkpoint directory up
# bash ~/projects/E2E/bash_training_scripts/evaluate_model_checkpoint.sh  \
#     -n $node \
#     -g $gpu \
#     -c $checkpoint_dir






