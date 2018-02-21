import argparse
import os
import csv

def main():
    parser = argparse.ArgumentParser(description='split TSV into source and'
                                     ' target files')
    parser.add_argument('-o', '--output_dir_name', help='output directory')
    parser.add_argument('-i', '--input_file_names', nargs='*', help='wikitext files')
    parser.add_argument('-t', '--test', help='test')
    args = parser.parse_args()

    for input_file_name in args.input_file_names:
        source = []
        target = []
        with open(input_file_name) as tsv:
            input_tsv = csv.reader(tsv, delimiter='\t')
            for line in input_tsv:
                source += [line[0]]
                if len(line) == 2:
                    target += [line[1]]
                elif len(line) > 2:
                    print('found too many tabbed columns: ', len(line))
                    return
        file_basename = os.path.basename(input_file_name)
        source_file_name = os.path.join(args.output_dir_name,
                                        file_basename + '.source')
        with open(source_file_name, 'w') as out:
            out.write('\n'.join(source))
        if target:
            target_file_name = os.path.join(args.output_dir_name,
                                            file_basename + '.target')
            with open(target_file_name, 'w') as out:
                out.write('\n'.join(target))

if __name__ == '__main__':
    main()
