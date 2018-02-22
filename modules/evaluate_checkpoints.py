import argparse
import os
import subprocess
import re

def main():
    parser = argparse.ArgumentParser(description='evaluate checkpoints using'
                                     ' bleu score')
    parser.add_argument('-o', '--output_dir_name', help='output directory')
    parser.add_argument('-i', '--input_file_names', nargs='*', help='model'
                        ' checkpoints')
    parser.add_argument('-s', '--source_file_name', help='source file')
    parser.add_argument('-t', '--target_file_name', help='target file')

    args = parser.parse_args()

    num_checkpoints_to_evaluate = 2

    checkpoint_accuracies = sorted([file_name.split('_acc_')[1] for file_name \
                         in args.input_file_names], reverse=True)
    if len(checkpoint_accuracies) > num_checkpoints_to_evaluate:
        num_extra = len(checkpoint_accuracies) - num_checkpoints_to_evaluate
        del checkpoint_accuracies[-num_extra:]
    # could probably do this using a list comprehension but whatever
    checkpoint_file_names = []
    for checkpoint_accuracy in checkpoint_accuracies:
        for input_file_name in args.input_file_names:
            if checkpoint_accuracy in input_file_name:
                checkpoint_file_names += [input_file_name]

    for checkpoint_file_name in checkpoint_file_names:
        file_root = os.path.splitext(os.path.basename(checkpoint_file_name))[0]
        # It won't always be true that this is using the validation file
        output_file_name = os.path.join(args.output_dir_name, file_root +
                                        '.valid.gen')
        translate_module_name = '/home/henrye/downloads/OpenNMT-py/translate.py'
        cmd = ['python', translate_module_name, '-model',
               checkpoint_file_name, '-gpu',
               os.environ['CUDA_VISIBLE_DEVICES'],
               '-src', args.source_file_name,
               '-output', output_file_name,
               '-dynamic_dict', '-share_vocab',
               '-batch_size', '1']
        print('translate checkpoint command: \n', ' '.join(cmd))
        subprocess.run(cmd)

        # TODO
        # There's definitely some more error handling stuff we could do here
        # based on what we found in this module
        # https://github.com/google/seq2seq/blob/master/seq2seq/metrics/bleu.py
        bleu_script_name =\
        '/home/henrye/downloads/OpenNMT-py/tools/multi-bleu.perl'
        cmd = ['perl', bleu_script_name, args.target_file_name]
        with open(output_file_name, 'r') as prediction_file:
            bleu_script_out = subprocess.check_output(cmd,
                                                      stdin=prediction_file).decode()
        bleu_score = re.search('BLEU = (.+?),', bleu_script_out).group(1)
        bleu_output_file_name =\
            os.path.join(os.path.split(output_file_name)[0], 'bleu_' +
                         bleu_score + '_' + os.path.split(output_file_name)[1])
        os.rename(output_file_name, bleu_output_file_name)

if __name__ == '__main__':
    main()
