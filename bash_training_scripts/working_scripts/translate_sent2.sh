
export CUDA_VISIBLE_DEVICES=1
BASE=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/data
MODEL=sent_train
MODEL_OG=struc_train
STEP=_step_6600
MODEL_DIR=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/model
TGEN=$BASE/tgen
OUTPUT=$BASE/output_$MODEL
OPENNMT=/home/nikhil/OpenNMT-py
GPU=0
TOTAL_GPUS=1


for model_checkpoint in $MODEL_DIR/$MODEL_OG$STEP.pt
    do
        echo $model_checkpoint
        python $OPENNMT/translate.py -model $model_checkpoint -src $OUTPUT-1 -gpu $GPU -beam_size 30 -batch_size 1 -max_length 500 -output $OUTPUT-final -n_best 1
#        cd /home/nikhil/Projects/e2e-metrics
#        conda activate Python2
        # measure scores run on python 2 so we switch back to the theano environment
#        python $EVAL_SCRIPT_DIR/measure_scores.py $EVAL_DATA_DIR/test-conc.txt $OUTPUT
#        conda deactivate
#        cd $BASE
    done
