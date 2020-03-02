import nltk
from pprint import pprint

# For NER validation using Wikipedia Articles
from wikipedia.wikipedia import search, page, suggest, summary

# Sentence extracted from Elon Musk's wikipedia page. 
sentence = '''
Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is an engineer, 
industrial designer, and technology entrepreneur.[2][3][4] He is a citizen of South Africa, 
Canada, and the United States (where he has lived most of his life and currently resides), 
and is the founder, CEO and chief engineer/designer of SpaceX;[5] co-founder,
CEO and product architect of Tesla, Inc.;[6][7] founder of The Boring Company;[8]
co-founder of Neuralink; and co-founder and initial co-chairman of OpenAI.[9] 
'''

# Generating the word tokens.
tokens = nltk.word_tokenize(sentence)

# Extraction of parts of speech from the tokens
pos_tags = nltk.pos_tag(tokens)

# Default Named Entity Recognition 
print(nltk.ne_chunk(pos_tags, binary=False))

# [RESULT]
# Tree('S', 
#         [Tree('PERSON', [('Elon', 'NNP')])
#         , Tree('PERSON', [('Reeve', 'NNP'), ('Musk', 'NNP'), ('FRS', 'NNP')])
#         , ('(', '(')
#         , ('/ˈiːlɒn/', 'NNP')
#         , (';', ':')
#         , ('born', 'VBN')
#         , ('June', 'NNP')
#         , ('28', 'CD')
#         , (',', ',')
#         , ('1971', 'CD')
#         , (')', ')')
#         , ('is', 'VBZ')
#         , ('an', 'DT')
#         , ('engineer', 'NN')
#         , (',', ',')
#         , ('industrial', 'JJ')
#         , ('designer', 'NN')
#         , (',', ',')
#         , ('and', 'CC')
#         , ('technology', 'NN')
#         , ('entrepreneur', 'NN')
#         , ('.', '.')
#         , ('[', 'CC')
#         , ('2', 'CD')
#         , (']', 'JJ')
#         , ('[', '$')
#         , ('3', 'CD')
#         , (']', 'NNP')
#         , ('[', 'VBD')
#         , ('4', 'CD')
#         , (']', 'NN')
#         , ('He', 'PRP')
#         , ('is', 'VBZ')
#         , ('a', 'DT')
#         , ('citizen', 'NN')
#         , ('of', 'IN')
#         , Tree('GPE', [('South', 'NNP'), ('Africa', 'NNP')])
#         , (',', ',')
#         , Tree('GPE', [('Canada', 'NNP')])
#         , (',', ',')
#         , ('and', 'CC')
#         , ('the', 'DT')
#         , Tree('GPE', [('United', 'NNP'), ('States', 'NNPS')])
#         , ('(', '(')
#         , ('where', 'WRB')
#         , ('he', 'PRP')
#         , ('has', 'VBZ')
#         , ('lived', 'VBN')
#         , ('most', 'JJS')
#         , ('of', 'IN')
#         , ('his', 'PRP$')
#         , ('life', 'NN')
#         , ('and', 'CC')
#         , ('currently', 'RB')
#         , ('resides', 'NNS')
#         , (')', ')')
#         , (',', ',')
#         , ('and', 'CC')
#         , ('is', 'VBZ')
#         , ('the', 'DT')
#         , ('founder', 'NN')
#         , (',', ',')
#         , Tree('ORGANIZATION', [('CEO', 'NNP')])
#         , ('and', 'CC')
#         , ('chief', 'JJ')
#         , ('engineer/designer', 'NN')
#         , ('of', 'IN')
#         , Tree('GPE', [('SpaceX', 'NNP')])
#         , (';', ':')
#         , ('[', 'VBZ')
#         , ('5', 'CD')
#         , (']', 'JJ')
#         , ('co-founder', 'NN')
#         , (',', ',')
#         , Tree('ORGANIZATION', [('CEO', 'NNP')])
#         , ('and', 'CC')
#         , ('product', 'NN')
#         , ('architect', 'NN')
#         , ('of', 'IN')
#         , Tree('GPE', [('Tesla', 'NNP')])
#         , (',', ',')
#         , Tree('GPE', [('Inc.', 'NNP')])
#         , (';', ':')
#         , ('[', 'VBD')
#         , ('6', 'CD')
#         , (']', 'NNP')
#         , ('[', 'VBD')
#         , ('7', 'CD')
#         , (']', 'JJ')
#         , ('founder', 'NN')
#         , ('of', 'IN')
#         , ('The', 'DT')
#         , Tree('ORGANIZATION', [('Boring', 'NNP'), ('Company', 'NNP')])
#         , (';', ':')
#         , ('[', 'VBD')
#         , ('8', 'CD')
#         , (']', 'JJ')
#         , ('co-founder', 'NN')
#         , ('of', 'IN')
#         , Tree('GPE', [('Neuralink', 'NNP')])
#         , (';', ':')
#         , ('and', 'CC')
#         , ('co-founder', 'NN')
#         , ('and', 'CC')
#         , ('initial', 'JJ')
#         , ('co-chairman', 'NN')
#         , ('of', 'IN')
#         , Tree('ORGANIZATION', [('OpenAI', 'NNP')])
#         , ('.', '.')
#         , ('[', 'CC')
#         , ('9', 'CD')
#         , (']', 'NN')]
#         )

# Extracting only Organizations and Geo Political Entities.
orgs_places = [" ".join([x[0] 
    for x in tree.leaves()]) 
        for tree in nltk.ne_chunk(pos_tags, binary=False)
            if isinstance(tree,nltk.Tree) 
            and tree.label() in ('ORGANIZATION', 'GPE')
]

print(orgs_places)

# Output

['South Africa'
    , 'Canada'
    , 'United States'
    , 'CEO'
    , 'SpaceX'
    , 'CEO'
    , 'Tesla'
    , 'Inc.'
    , 'Boring Company'
    , 'Neuralink'
    , 'OpenAI']

# Regular Expression based chunking and entities extraction.

grammar = r"""
    NP: {<JJ|NN.*>+}          # Chunk sequences of JJ, NN
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}     
  """

chunk_parser = nltk.RegexpParser(grammar)
entities = [(" ".join(
                        [x[0] for x in tree.leaves()]), tree.label()) 
                            for tree in chunk_parser.parse(pos_tags) 
                            if isinstance(tree,nltk.Tree)]

pprint(entities)

# Output
[('Elon Reeve Musk FRS', 'NP'), ('/ˈiːlɒn/', 'NP'),
 ('June', 'NP'), ('engineer', 'NP'),
 ('industrial designer', 'NP'),
 ('technology entrepreneur', 'NP'),
 (']', 'NP'),
 (']', 'NP'),
 (']', 'NP'),
 ('citizen', 'NP'),
 ('South Africa', 'NP'),
 ('Canada', 'NP'),
 ('United States', 'NP'),
 ('life', 'NP'),
 ('resides', 'NP'),
 ('founder', 'NP'),
 ('CEO', 'NP'),
 ('chief engineer/designer', 'NP'),
 ('SpaceX', 'NP'),
 ('] co-founder', 'NP'),
 ('CEO', 'NP'),
 ('product architect', 'NP'),
 ('Tesla', 'NP'),
 ('Inc.', 'NP'),
 (']', 'NP'),
 ('] founder', 'NP'),
 ('Boring Company', 'NP'),
 ('] co-founder', 'NP'),
 ('Neuralink', 'NP'),
 ('co-founder', 'NP'),
 ('initial co-chairman', 'NP'),
 ('OpenAI', 'NP'),
 (']', 'NP')]
