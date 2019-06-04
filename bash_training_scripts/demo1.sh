
export CUDA_VISIBLE_DEVICES=1
BASE=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/data
MODEL=baseline_test_gpu_0
STEP=_step_5000
MODEL_DIR=/home/nikhil/Projects/E2E-NLG-Challenge/E2E/model
TGEN=$BASE/tgen
OUTPUT=$BASE/output
OPENNMT=/home/nikhil/OpenNMT-py
GPU=0
TOTAL_GPUS=1


for model_checkpoint in $MODEL_DIR/$MODEL$STEP.pt
    do
        echo $model_checkpoint
        python $OPENNMT/translate.py -model $model_checkpoint -src $BASE/devset-source-grouped.tok -gpu $GPU -beam_size 30 -batch_size 1 -max_length 500 -output $OUTPUT -n_best 1
#        cd /home/nikhil/Projects/e2e-metrics
#        conda activate Python2
        # measure scores run on python 2 so we switch back to the theano environment
#        python measure_scores.py $TGEN/devset-target.grouped $OUTPUT
#        conda deactivate
#        cd /home/nikhil/OpenNMT-py
    done
