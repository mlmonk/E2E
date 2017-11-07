checkpoint_dir=/home/henrye/projects/E2E/experiments/features-layers,dropout,vec_sizes-27893-2017_10_31__18_30/checkpoints-boole-n021-gpu_1-2017_11_01__08_36/
checkpoint_name=$(basename $checkpoint_dir)
special_bash_script=~/projects/E2E/bash_training_scripts/create_sample_outputs_for_this_checkpoint_dir_$checkpoint_name.sh
rm -f $special_bash_script
touch $special_bash_script
echo "rm -f /home/henrye/projects/E2E/experiments/features-layers,dropout,vec_sizes-27893-2017_10_31__18_30/lots_and_lots_of_sample_outputs_from_$checkpoint_name.txt" >> $special_bash_script
echo "paste -d '\n' \\" >> $special_bash_script
for temperature in normies temperature_{0.7,0.8,0.9,1.0,1.1}
do
    echo /home/henrye/projects/E2E/experiments/features-layers,dropout,vec_sizes-27893-2017_10_31__18_30/test*$temperature \\ >> $special_bash_script
    echo $checkpoint_dir/*/*test*$temperature* \\ >> $special_bash_script
done
# we add the first file back in again because this brings it up to a round number, 25, which makes investigating much easier
echo "/home/henrye/projects/E2E/experiments/features-layers,dropout,vec_sizes-27893-2017_10_31__18_30/test_e2e-source.tok.normies \\" >> $special_bash_script
echo "> /home/henrye/projects/E2E/experiments/features-layers,dropout,vec_sizes-27893-2017_10_31__18_30/lots_and_lots_of_sample_outputs_from_$checkpoint_name.txt" >> $special_bash_script
bash $special_bash_script
