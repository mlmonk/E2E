import argparse
from sigopt import Connection
import glob
import re
import subprocess
import os



parser = argparse.ArgumentParser(description='training')
parser.add_argument('-node', default=None, help='node to use, in format boole-n021')
parser.add_argument('-gpu', default='0', help='which gpu to use')
parser.add_argument('-experiment_id', default=None, help='the folder containing the preprocessed data')
opts = parser.parse_args()

def main():
    if not opts.node or not opts.experiment_id:
        print('please supply all the command line inputs')
        quit()

    conn = Connection(client_token="IFAQZABYDOBABXMSZYAWSYKHYSONHNPEACATCSCCIDXDQFLG")
    
    # start for loop
    for i in range(20):
        suggestion = conn.experiments(opts.experiment_id).suggestions().create(
            metadata={
                'node': opts.node,
                'gpu': str(opts.gpu)
                }
            )
        training_opts = []
        # this is kind of a messy workaround because opennmt-py doens't like
        # vector sizes with odd numbers
        for key, value in suggestion.assignments.items():
            if 'rnn_size' in key or 'word_vec_size' in key:
                value = value * 2
            if 'dropout' in key:
                value = "{0:.2f}".format(value)
            training_opts += [key] + [str(value)]
        training_opts = ' '.join(training_opts)
        training_opts = '\"' + training_opts + '\"'
        preprocessed_dir = \
            glob.glob('/home/henrye/projects/E2E/experiments/*' + opts.experiment_id + '*')[0]
        print('starting training with suggestion: ', training_opts)
        training_output_full = subprocess.run(['bash',
            '/home/henrye/projects/E2E/bash_training_scripts/start_training_template.sh',
            '-n', opts.node,
            '-g', opts.gpu,
            '-p', preprocessed_dir,
            '-t', training_opts],
                      stdout=subprocess.PIPE)
        print('here is the bash command for testing training:\n', ' '.join(training_output_full.args))                  
        print('finished training, starting evaluation')
        training_output = training_output_full.stdout.decode()
        checkpoint_dir = re.search('the checkpoint directory is (.*) end transmission', \
                                   training_output).group(1)
        evaluation_opts = 'empty'
        evaluation_output_full = subprocess.run(['bash',
        '/home/henrye/projects/E2E/bash_training_scripts/evaluate_model_checkpoint.sh',
        '-n', opts.node,
        '-g', opts.gpu,
        '-c', checkpoint_dir,
        '-e', evaluation_opts],
                  stdout=subprocess.PIPE)
        print('here is the bash command for testing evaluation:\n', ' '.join(evaluation_output_full.args))
        evaluation_output = evaluation_output_full.stdout.decode()
        generated_outputs_directory = re.search('the generated outputs directory is (.*) end transmission', \
                                   evaluation_output).group(1)
        print('outputs directory: ', generated_outputs_directory, 
            '\n there should be familiar looking files in here')
        output_scores = glob.glob(os.path.join(generated_outputs_directory, 'BLEU*dev*'))
        output_scores = sorted(output_scores, reverse=True)
        best_bleu = re.search('BLEU,(.*),NIST', output_scores[0]).group(1)

        # submit bleu in observation

        print('best bleu: ', best_bleu)
        conn.experiments(opts.experiment_id).observations().create(
            suggestion=suggestion.id,
            value=float(best_bleu),
            metadata={
                'node': opts.node,
                'gpu': str(opts.gpu),
                'output_dir': generated_outputs_directory
            }
        )



if __name__ == '__main__':
    main()