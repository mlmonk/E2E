import spacy

nlp = spacy.load('en')

for token1 in tokens:
        for token2 in tokens:
                    print(token1.similarity(token2))

sent1 = nlp('Near The Six Bells is a Fitzbillies , providing cheap Italian food .')
sent2 = nlp('The Dumpling Tree is an Indian restaurant with cheap prices')
sent3 = nlp('In the riverside area is located a child friendly coffee shop .  Prices are in the £ 20 - 25 range and the shop has a high customer rating , look for The Zizzi .')
for sent in [sent1, sent2, sent3]:
    for other_sent in [sent1, sent2, sent3]:
        print(sent, '\n', other_sent, '\n', sent.similarity(other_sent))

for token in sent1:
        print(token.text, token.has_vector, token.vector_norm, token.is_oov)

with open('/home/henrye/projects/E2E/experiments/end_to_end_test_2018-02-21_11-08_38603/create_source_target/jess_2018-02-23_07-27/devset.tok.target', 'r') as target_sents:
    sents = target_sents.readlines()

sents = [nlp(sent) for sent in sents]
similarities = {}
for i, sent in enumerate(sents):
    similarities[sents[0].similarity(sent)] = i

this = sorted(list(similarities.keys()), reverse=True)
[print(sents[similarities[this_key]]) for this_key in this[0:10]]
# nlp('Punter')[0].is_oov
