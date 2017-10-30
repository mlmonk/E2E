#!/bin/bash
# TODO investigate when to use quotes for 
# This function will create a folder with preprocessed data in it and can be 
# called from anywhere. We can probably turn it into a function that we source
# in bashrc and then when we want to create a new folder for baseline training
# just called the function with baseline or features command line option

# Usage - preferably run inside a node
# bash create_experiment_dir.sh baseline

# What will it do
# create director, preprocess files and copy over files for evaluation


# for additional viewing and debugging pleasure
# set -x

# Usage
# bash ~/projects/E2E/bash_training_scripts/create_experiment_dir.sh \
#     -p 'layers,dropout,vec_sizes' \
#     -t baseline

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -p|--model_parameters)
    model_parameters="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--model_type)
    model_type="$2"
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

if [ $model_type != features ] && [ $model_type != baseline ]
    then
    echo "wrong model type option used. Only features or baseline are valid"
    exit 1
fi

if [ $model_parameters != test ] && [ $model_parameters != 'layers,dropout,vec_sizes' ]
    then
    echo "only test or 'layers,dropout,vec_sizes' allowed right now"
    exit 1
fi

data_dir=/home/henrye/projects/E2E/data
experiment_dir=/home/henrye/projects/E2E/experiments
now=`date '+%Y_%m_%d__%H_%M'`

source activate pytorch
experiment_id=`python ~/projects/E2E/initialise_experiment_id.py \
                        -parameters $model_parameters`
model_dir="$experiment_dir/$model_type-$model_parameters-$experiment_id-$now"
mkdir $model_dir

current_dir=`pwd`
cd ~/downloads/OpenNMT-py
# TODO we're not sure how to limit the depth of set -x so we're just going 
# ignore this command instead
if [ $model_type == baseline ]
    then
    python preprocess.py \
    -train_src $data_dir/trainset-source.tok  \
    -train_tgt $data_dir/trainset-target.tok \
    -valid_src $data_dir/devset-source.tok \
    -valid_tgt $data_dir/devset-target.tok \
    -save_data $model_dir/preprocessed \
    -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab
    # we copy over what we assume to be canonical evaluation source files
    cp $data_dir/{devset-source.tok.unique,test_e2e-source.tok} $model_dir/
elif [ $model_type == features ]
    then
    python preprocess.py \
    -train_src $data_dir/trainset-source.tok.extracted_feat \
    -train_tgt $data_dir/trainset-target.tok \
    -valid_src $data_dir/devset-source.tok.extracted_feat \
    -valid_tgt $data_dir/devset-target.tok \
    -save_data $model_dir/preprocessed \
    -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab 
    # we copy over what we assume to be canonical evaluation source files
    # TODO we haven't changed these to include the features yet
    touch $model_dir/.WARNING_these_source_files_do_not_have_additional_words
    cp $data_dir/{devset-source.tok.unique,test_e2e-source.tok} $model_dir/
fi
cd $current_dir



# TODO add in the features training option
# TODO eventually we may care about copying the underlying source and target files for posterity and reproducibility

