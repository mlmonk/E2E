import argparse
import os
import csv
from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize import sent_tokenize


word_tok = MosesTokenizer()
def tokenize(text):
    word_tokens = word_tok.tokenize(text, agressive_dash_splits=True,
                                    escape=False)
    return word_tokens

def main():
    parser = argparse.ArgumentParser(description='Command-line script for separating wikitext in a list of sentences')
    parser.add_argument('-o', '--output_dir_name', help='output directory')
    parser.add_argument('-i', '--input_file_names', nargs='*', help='wikitext files')
    parser.add_argument('-t', '--test', help='test')
    args = parser.parse_args()

    for input_file_name in args.input_file_names:
        input_csv = csv.reader(open(input_file_name, newline=''),
                               delimiter=',', quotechar='"')
        next(input_csv)
        output = []
        for line in input_csv:
            meaning_representations = line[0].split(', ')
            acts_tok = []
            for act in meaning_representations:
                act_type = act[0:act.find('[')].replace(' ', '')
                acts_tok += ['__start_' + act_type + '__']
                acts_tok += tokenize(act[act.find('[')+1:act.find(']')])
                acts_tok += ['__end_' + act_type + '__']
            acts = ' '.join(acts_tok).strip().replace('\@-\@', '-')
            if len(line) == 2:
                target_tok = [tokenize(t) for t in sent_tokenize(line[1])]
                target = ' '.join([j for i in target_tok for j in
                                   i]).strip().replace('\@-\@', '-')
                this_output = '\t'.join([acts, target])
            elif len(line) > 2:
                print('csv reader has found too many columns: ', len(line))
                return
            else:
                this_output = acts
            output += [this_output]

        file_basename = os.path.basename(input_file_name)
        file_root = os.path.splitext(file_basename)[0]
        output_file_basename = file_root + '.tok'
        output_file_name = os.path.join(args.output_dir_name,
                                        output_file_basename)
        with open(output_file_name, 'w') as out_file:
            out_file.write('\n'.join(output))

if __name__ == '__main__':
    main()
