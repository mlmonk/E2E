
export CUDA_VISIBLE_DEVICES=1
BASE=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/data
MODEL=struc_train
STEP=_step_6600
MODEL_DIR=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/model
EVAL_DATA_DIR=$BASE/evaluation_dataset
OUTPUT=$BASE/output_$MODEL
OPENNMT=/home/nikhil/OpenNMT-py
EVAL_SCRIPT_DIR=/home/nikhil/Projects/e2e-metrics
GPU=0
TOTAL_GPUS=1


for model_checkpoint in $MODEL_DIR/$MODEL$STEP.pt
    do
        echo $model_checkpoint
        python $OPENNMT/translate.py -model $model_checkpoint -src $BASE/testset-struc-source.tok -gpu $GPU -beam_size 30 -batch_size 1 -max_length 500 -output $OUTPUT -n_best 1
#        cd /home/nikhil/Projects/e2e-metrics
#        conda activate Python2
        # measure scores run on python 2 so we switch back to the theano environment
#        python $EVAL_SCRIPT_DIR/measure_scores.py $EVAL_DATA_DIR/test-conc.txt $OUTPUT
#        conda deactivate
#        cd $BASE
    done
