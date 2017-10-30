#!/bin/bash

# Usage
# bash ~/projects/E2E/bash_training_scripts/evaluate_model_checkpoint.sh  \
#     -n boole-n021 \
#     -g 0 \
#     -c /home/henrye/projects/E2E/experiments/baseline-test-27604-2017_10_29__15_03/checkpoints-boole-n021-gpu_0-2017_10_30__13_20/

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
    -c|--checkpoint_dir)
    checkpoint_dir="$2"
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

# get the parent directory of checkpoint_dir
    # this will give us access to the devset and testset files
preprocessed_dir="$(dirname "$checkpoint_dir")"

# make a new evaluation folder for -output
now=$(date '+%Y_%m_%d__%H_%M')
generated_outputs_dir="$checkpoint_dir/generated-outputs-$node-gpu_$gpu-$now"
if [ ! -d "$generated_outputs_dir" ]; then
    mkdir "$generated_outputs_dir"
else
    echo "directory already exists, you must be doing testing"
    generated_outputs_dir="$generated_outputs_dir-test"
    mkdir $generated_outputs_dir
fi

# We need to add backslashes before all the variables assigned during ssh
# https://stackoverflow.com/questions/13032409/ssh-remote-variable-assignment
ssh -t -t $node << EOF
    set -x
    echo 'the generated outputs directory is $generated_outputs_dir end transmission'
    # We've decided to just evaluate epoch 20 for quickness, might change later
    for model_checkpoint in $checkpoint_dir/checkpoint*e{20..21}.pt
    do
        if [ ! -f \$model_checkpoint ]; then
            echo "file not found"
            continue
        fi
        echo \$model_checkpoint
        source activate pytorch
        cd ~/downloads/OpenNMT-py
        model_checkpoint_file_name=\$(basename "\$model_checkpoint")
        for source_file in $preprocessed_dir/{dev*,test*}
        do
            source_file_name=\$(basename "\$source_file")
            output_file_name=\$source_file_name-\$model_checkpoint_file_name

            # running tracer output for testing
            # echo \$output_file_name
            # touch $generated_outputs_dir/\$output_file_name

            # for non tracer output this is where we run python translate
            python translate.py \
                -model \$model_checkpoint \
                -src \$source_file \
                -output $generated_outputs_dir/\$output_file_name \
                -gpu $gpu \
                -max_sent_length 500 \
                -beam_size 30 \
                -batch_size 1 \
                -n_best 1
        done
        source activate theano
        cd /home/henrye/downloads/e2e-metrics

        # test scores for tracer program
        # checkpoint_scores="BLEU,0.6537,NIST,7.7660,METEOR,0.4819,ROUGE_L,0.7319,CIDEr,2.3479"

        # in the non tracer output this is where we run the python measure_scores
        # function. which is what assume to be a set of steps for correctly
        # extracting the scores
        checkpoint_scores=\$(python measure_scores.py \
            /home/henrye/projects/E2E/data/devset-target.grouped \
            $generated_outputs_dir/dev*\$model_checkpoint_file_name)
        checkpoint_scores=\$(echo \$checkpoint_scores | grep -P -o "= \K.*")
        checkpoint_scores=\${checkpoint_scores//\ /,}
        checkpoint_scores=\${checkpoint_scores//:/}

        echo \$checkpoint_scores

        for generated_outputs in $generated_outputs_dir/{dev,test}*\$model_checkpoint_file_name
        do
            echo \$generated_outputs
            mv \$generated_outputs \${generated_outputs/\$model_checkpoint_file_name/\$checkpoint_scores-\$model_checkpoint_file_name}
        done

    done
    exit
EOF