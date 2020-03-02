import warnings

import pandas as pd
import spacy
from spacy.tokens import Doc, Token, Span

warnings.filterwarnings("ignore") 

nlp = spacy.load("en_core_web_sm")

# Few doc strings to experiment with
doc = nlp("Apple isn't looking at buying U.K. startup for $1 billion")
doc = nlp("I like cats")
doc = nlp("I like dogs more than cats")

doc = nlp("This is as cold as an unused pillow")

# Ridiculously long string for speed test.
doc = nlp("""The rainbow pitta (Pitta iris) is a small passerine bird in the pitta family, Pittidae, endemic to northern Australia. The species is most closely related to the superb pitta of Manus Island. A colourful bird, it has a velvet black head with chestnut stripes above the eyes, olive green upper parts, black underparts, a bright red belly and an olive green tail. The rainbow pitta lives in the monsoon forests, as well as some drier eucalypt forests. As with other pittas, it is a secretive and shy bird. The diet consists mainly of insects, arthropods and small vertebrates. Pairs defend territories and breed during the rainy season, as this time of year provides the most food for nestlings. The female lays three to four eggs with blotches inside its large domed nest. Both parents defend the nest, incubate the eggs and feed the chicks. The species is common within its range, and is not threatened. (Full article...)""")


# Tablulating the results for better understanding
doc_items_df = pd.DataFrame(columns=["text", "lemma_", "pos_", "des_pos", "tag_", "des_tag", "dep_", "des_dep", "shape_", "is_alpha", "is_stop"])

for token in doc:
    doc_items_df = doc_items_df.append(
                    pd.DataFrame([[
                                                token.text
                                                , token.lemma_
                                                , token.pos_
                                                , spacy.explain(token.pos_)
                                                , token.tag_
                                                , spacy.explain(token.tag_)
                                                , token.dep_
                                                , spacy.explain(token.dep_)
                                                , token.shape_
                                                , token.is_alpha
                                                , token.is_stop
                                            ]
                                        ]
                                    , columns = doc_items_df.columns)
                        )
doc_items_df.reset_index(drop = True, inplace = True)
print(doc_items_df)


"""

       text   lemma_   pos_      des_pos tag_                                    des_tag      dep_                    des_dep shape_ is_alpha is_stop
0     Apple    Apple  PROPN  proper noun  NNP                      noun, proper singular     nsubj            nominal subject  Xxxxx     True   False
1        is       be    AUX    auxiliary  VBZ          verb, 3rd person singular present       aux                  auxiliary     xx     True    True
2       n't      not   PART     particle   RB                                     adverb       neg          negation modifier    x'x    False    True
3   looking     look   VERB         verb  VBG         verb, gerund or present participle      ROOT                       None   xxxx     True   False
4        at       at    ADP   adposition   IN  conjunction, subordinating or preposition      prep     prepositional modifier     xx     True    True
5    buying      buy   VERB         verb  VBG         verb, gerund or present participle     pcomp  complement of preposition   xxxx     True   False
6      U.K.     U.K.  PROPN  proper noun  NNP                      noun, proper singular  compound                   compound   X.X.    False   False
7   startup  startup   NOUN         noun   NN                     noun, singular or mass      dobj              direct object   xxxx     True   False
8       for      for    ADP   adposition   IN  conjunction, subordinating or preposition      prep     prepositional modifier    xxx     True    True
9         $        $    SYM       symbol    $                           symbol, currency  quantmod     modifier of quantifier      $    False   False
10        1        1    NUM      numeral   CD                            cardinal number  compound                   compound      d    False   False
11  billion  billion    NUM      numeral   CD                            cardinal number      pobj      object of preposition   xxxx     True   False


Description of the columns: 
---------------------------
Text: The original word text.
Lemma: The base form of the word.
POS: The simple part-of-speech tag.
Tag: The detailed part-of-speech tag.
Dep: Syntactic dependency, i.e. the relation between tokens.
Shape: The word shape – capitalization, punctuation, digits.
is alpha: Is the token an alpha character?
is stop: Is the token part of a stop list, i.e. the most common words of the language?

"""

ent_df = pd.DataFrame(columns=["Entity", "Start", "End", "Tag", "Tag_Desciption"])
for ent in doc.ents:
    # print(ent.text, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_))
    ent_df = ent_df.append(pd.DataFrame(
                                [[ent.text, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_)]]
                                , columns = ent_df.columns))
print(ent_df)

"""
       Entity Start End    Tag                           Tag_Desciption
0       Apple     0   5    ORG  Companies, agencies, institutions, etc.
0        U.K.    30  34    GPE                Countries, cities, states
0  $1 billion    47  57  MONEY          Monetary values, including unit

Column Description:
-----------------------------------
Start = Index of the first letter in doc
End = Index of the last letter in doc

"""


# Understanding relationship between words in sentences
doc_1 = nlp("I like swimming")
doc_2 = nlp("I like cycling more than driving")

matches_df = pd.DataFrame(columns= ["token1", "token2", "similarity"])
for token1 in doc_1:
    for token2 in doc_2:
        if token1 != token2 and not (token1.is_stop or token2.is_stop):
            matches_df = matches_df.append(
                pd.DataFrame([[token1, token2, token1.similarity(token2)]], columns = matches_df.columns)
            )
            # print(token1, token2, token1.similarity(token2))

matches_df.sort_values("similarity", ascending = False).iloc[::2]









