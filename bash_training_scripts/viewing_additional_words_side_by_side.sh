
if [ ! -f /home/henrye/projects/E2E/experiments/additional_words_generator-2017_10_31__09_32/side_by_side_dev_outputs.txt ]
    then
    echo $(dirname checkpoints-boole-n023-gpu_*10_02/gen*_{0.7,0.8,0.9,1.0,1.1}_*beam_size_1-*/dev*) > /home/henrye/projects/E2E/experiments/additional_words_generator-2017_10_31__09_32/side_by_side_dev_outputs.txt
    paste -d ',' checkpoints-boole-n023-gpu_*10_02/gen*_{0.7,0.8,0.9,1.0,1.1}_*beam_size_1-*/dev* >> /home/henrye/projects/E2E/experiments/additional_words_generator-2017_10_31__09_32/side_by_side_dev_outputs.txt
fi