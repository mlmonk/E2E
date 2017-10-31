generated_additional_words_dir=/home/henrye/projects/E2E/experiments/additional_words_generator-2017_10_31__09_32/checkpoints-boole-n023-gpu_1-2017_10_31__10_02

for source in /home/henrye/projects/E2E/data/devset-source.tok.unique /home/henrye/projects/E2E/data/test_e2e-source.tok
do
    for temperature in temperature_{0.7,0.8,0.9,1.0,1.1}
    do
        if [[ $source == *"dev"* ]]; then
            echo dev
            dev_or_test=dev
        else
            dev_or_test=test
            echo test
            echo $source.gen_feat.$temperature 
        fi
        echo "creating file: $source.gen_feat.$temperature"
        paste -d '|' \
            $source \
            $generated_additional_words_dir/*$temperature*beam_size_1*/$dev_or_test* \
            | sed "s/|/ __start_additional_words__ /g" \
            | sed "s/$/ __end_additional_words__/" \
            | sed "s/__start_additional_words__  __end_additional_words__//" \
            > $source.gen_feat.$temperature
    done
done