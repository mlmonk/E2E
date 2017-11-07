#!/bin/bash

# Usage
# bash ~/projects/E2E/bash_training_scripts/evaluate_model_checkpoint.sh  \
#     -n boole-n023 \
#     -g 0 \
#     -c /home/henrye/projects/E2E/experiments/baseline-layers,dropout,vec_sizes-27868-2017_10_30__15_36/checkpoints-boole-n021-gpu_0-2017_10_31__12_23
evaluation_opts="-beam_size 30"

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

if [ $evaluation_opts = 'empty' ]
    then
    # TODO This is a default evaluation value but it would be better for the 
    # training loop to supply its own default value
    evaluation_opts="-beam_size 30"
fi

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
    for model_checkpoint in $checkpoint_dir/checkpoint*e{7..9}.pt
    do
        if [ ! -f \$model_checkpoint ]; then
            echo "file not found"
            continue
        fi
        echo \$model_checkpoint
        set +x; source activate pytorch; set -x
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
                -batch_size 1 \
                -beam_size 30 \
                -n_best 1
        done
        set +x; source activate theano; set -x
        cd /home/henrye/downloads/e2e-metrics

        for translated_dev_file in $generated_outputs_dir/dev*\$model_checkpoint_file_name
        do
            # fake checkpoint scores, otherwise do the same python function, with translated_dev_file
            # as the input rather than the dev* thing we have at the moment
            # checkpoint_scores="BLEU,0.6537,NIST,7.7660,METEOR,0.4819,ROUGE_L,0.7319,CIDEr,2.3479"

            # in the non tracer output this is where we run the python measure_scores
            # function. which is what assume to be a set of steps for correctly
            # extracting the scores
            checkpoint_scores=\$(python measure_scores.py \
                /home/henrye/projects/E2E/data/devset-target.grouped \
                \$translated_dev_file)
            checkpoint_scores=\$(echo \$checkpoint_scores | grep -P -o "= \K.*")
            checkpoint_scores=\${checkpoint_scores//\ /,}
            checkpoint_scores=\${checkpoint_scores//:/}

            temperature_checkpoint_name=\$(basename "\$translated_dev_file")
            temperature_checkpoint_name=\${temperature_checkpoint_name/"devset-source.tok.unique"/}
            echo \$temperature_checkpoint_name
            for this_temperature_checkpoint_output in $generated_outputs_dir/*\$temperature_checkpoint_name
            do
                file_name=\$(basename "\$this_temperature_checkpoint_output")
                echo \${this_temperature_checkpoint_output/\$file_name/\$checkpoint_scores-\$file_name}
                mv \$this_temperature_checkpoint_output \${this_temperature_checkpoint_output/\$file_name/\$checkpoint_scores-\$file_name}
            done
        done


 

        # echo \$checkpoint_scores

        # for generated_outputs in $generated_outputs_dir/{dev,test}*\$model_checkpoint_file_name
        # do
        #     echo \$generated_outputs
        #     mv \$generated_outputs \${generated_outputs/\$model_checkpoint_file_name/\$checkpoint_scores-\$model_checkpoint_file_name}
        # done

    done
    exit
EOF