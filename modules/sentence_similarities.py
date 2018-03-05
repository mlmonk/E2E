import argparse
import os
import shutil
import re
import math
import spacy
import Levenshtein
import json
from collections import defaultdict, Counter

def main():
    parser = argparse.ArgumentParser(description='Find simlar sentences')
    parser.add_argument('-o', '--output_dir_name', help='output directory')
    parser.add_argument('--train_src_name')
    parser.add_argument('--train_tgt_name')
    parser.add_argument('--logging_dir_name', help='logging directory')
    parser.add_argument('--similarity_metric', choices=['spacy', 'lev',
                                                        'cosine'],
                        help='sentence similarity metrc')
    parser.add_argument('--extra_filter', action='store_true', help='filter the'
                        ' additional MRs; eatType, food, area')
    args = parser.parse_args()

    sub_sample = 5

    with open(args.train_src_name, 'r') as in_file:
        source_mrs = [line.strip() for line in in_file]
    with open(args.train_tgt_name, 'r') as in_file:
        target_sents = [line.strip() for line in in_file]

    filter_words = set()
    for mrs in source_mrs:
        # TODO
        # This seems like something that could be reduced to a function rather
        # than repeated five times
        possible_name = re.search('__start_name__ (.+?) __end_name__', mrs)
        if possible_name:
            filter_words.add(possible_name.group(1))
        possible_near = re.search('__start_near__ (.+?) __end_near__', mrs)
        if possible_near:
            filter_words.add(possible_near.group(1))
        if args.extra_filter:
            possible_food = re.search('__start_food__ (.+?) __end_food__', mrs)
            if possible_food:
                filter_words.add(possible_food.group(1))
            possible_area = re.search('__start_area__ (.+?) __end_area__', mrs)
            if possible_area:
                filter_words.add(possible_area.group(1))
            possible_eatType = re.search('__start_eatType__ (.+?)'
                                         '__end_eatType__', mrs)
            if possible_eatType:
                filter_words.add(possible_eatType.group(1))

    filter_words_compiled = [re.compile(re.escape(filter_word), re.IGNORECASE) \
                             for filter_word in filter_words]

    numbers_as_words = ['one', 'two', 'three', 'four', 'five']
    sents_oov = []
    for sent in target_sents:
        for filter_word in filter_words_compiled:
            sent = filter_word.sub('￨', sent)
        for tok in sent.split(' '):
            if any(char.isdigit() for char in tok) or tok.lower() in \
                                                        numbers_as_words:
                sent = sent.replace(tok, '￨')
        sents_oov += [sent]

    # TODO
    # Put inside a big if statement and check if there already exists a
    # matching variable in the json file. If not then do the calculations and
    # save the similar_sents_idx to the json with the similarity arg as the key
    precalc_similar_sents = defaultdict(list)
    similar_sents_idxs = []
    precalc_similar_sents_file_name = os.path.join(args.logging_dir_name,
                                               'precalc_similar_sents.json')
    similarity_filter_key = args.similarity_metric
    if os.path.isfile(precalc_similar_sents_file_name):
        with open(precalc_similar_sents_file_name) as in_file:
            precalc_similar_sents = json.load(in_file)
        if similarity_filter_key in precalc_similar_sents:
            similar_sents_idxs = precalc_similar_sents[similarity_filter_key]
    # We want to save on the computation of the similar sentences since it can
    # take quite a while
    if not similar_sents_idxs:
        if args.similarity_metric == 'lev':
            sents_oov_extra = sents_oov
            find_max = False
        elif args.similarity_metric == 'spacy':
            nlp = spacy.load('en')
            sents_oov_extra = [nlp(sent) for sent in sents_oov]
            find_max = True
        elif args.similarity_metric == 'cosine':
            sents_oov_extra = [text_to_vector(sent) for sent in sents_oov]
            find_max = True

        for ref_sent in sents_oov_extra[0:sub_sample]:
            similarities = defaultdict(list)
            for sent_idx, compar_sent in enumerate(sents_oov_extra):
                if args.similarity_metric == 'lev':
                    similarity_value = Levenshtein.distance(ref_sent, compar_sent)
                elif args.similarity_metric == 'spacy':
                    similarity_value = ref_sent.similarity(compar_sent)
                elif args.similarity_metric == 'cosine':
                    similarity_value = get_cosine(ref_sent, compar_sent)
                similarities[similarity_value].append(sent_idx)
            similarities_sorted = [similarities[k] for k in \
                                   sorted(similarities, key=similarities.get,
                                          reverse=find_max)]
            # flatten the list of lists
            similarities_sorted = [y for x in similarities_sorted for y in x]
            # if we stored all 42,000 rankings for each of the 42,000 sentences
            # the file would be way too big. Top 10 should be plenty
            similar_sents_idxs.append(similarities_sorted[0:11])
        precalc_similar_sents[similarity_filter_key] = similar_sents_idxs
        with open(precalc_similar_sents_file_name, 'w') as out_file:
            json.dump(precalc_similar_sents, out_file)

    similar_sents = []
    similar_sents_oov = []
    for sent_idx in range(sub_sample):
        similar_sents.append('\t'.join([target_sents[idx] for idx in \
                                        similar_sents_idxs[sent_idx][0:6]]))
        similar_sents_oov.append('\t'.join([sents_oov[idx] for idx in \
                                        similar_sents_idxs[sent_idx][0:6]]))

    file_basename = os.path.basename(args.train_src_name)
    output_file_basename = file_basename + '.similar_sents_' + \
                           args.similarity_metric
    output_file_name = os.path.join(args.output_dir_name, output_file_basename)
    with open(output_file_name, 'w') as out_file:
        out_file.write('\n'.join(similar_sents))
    file_basename = os.path.basename(args.train_src_name)
    output_file_basename = file_basename + '.similar_sents_oov_' + \
                           args.similarity_metric
    output_file_name = os.path.join(args.output_dir_name, output_file_basename)
    with open(output_file_name, 'w') as out_file:
        out_file.write('\n'.join(similar_sents_oov))
    # TODO
    # If not debug then add N similar sentences to the source MRs and do this M
    # times. Make sure there are M copies of each target sent in the
    # train_target output file

    # source_mrs_with_sents = ['{}\t{}'.format(*combined) for combined in \
    #                          zip(source_mrs, similar_sents)]
    # output_file_basename = os.path.basename(args.train_tgt_name)
    # output_file_name = os.path.join(args.output_dir_name, output_file_basename)
    # shutil.copy(args.train_tgt_name, output_file_name)

    # similarities = {}
    # if args.similarity_metric == 'lev':
    #     for i, sent in enumerate(sents_oov):
    #         similarities[Levenshtein.distance(sents_oov[0], sent)] = i
    #     similarities_sorted = sorted(list(similarities.keys()))
    # elif args.similarity_metric == 'spacy':
    #     nlp = spacy.load('en')
    #     sents_oov_nlp = [nlp(sent) for sent in sents_oov]
    #     for i, sent in enumerate(sents_oov_nlp):
    #         similarities[sents_oov_nlp[0].similarity(sent)] = i
    #     similarities_sorted = sorted(list(similarities.keys()), reverse = True)
    # similar_sents = '\t'.join([target_sents[similarities[this_key]] \
    #                          for this_key in similarities_sorted])


    # this = sorted(list(similarities.keys()), reverse=True)
    # [print(target_sents[similarities[this_key]]) for this_key in this[0:10]]
    # # nlp('Punter')[0].is_oov

# one_word = re.compile('\w+')
# bigrams = re.compile(' (?=((?:\w+|\W) (?:\w+|\W)))')

# def get_cosine(vec1, vec2):
#     intersection = set(vec1.keys()) & set(vec2.keys())
#     numerator = sum([vec1[x] * vec2[x] for x in intersection])
#     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
#     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
#     denominator = math.sqrt(sum1) * math.sqrt(sum2)
#     if not denominator:
#         return 0.0
#     else:
#         return float(numerator) / denominator

# def text_to_vector(text):
#     words = one_word.findall(text)
#     return Counter(words)

if __name__ == '__main__':
    main()
