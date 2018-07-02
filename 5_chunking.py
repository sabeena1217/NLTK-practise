## name entity - noun

##chunks of noun phrases, basically grouping of nouns


'''chinking

removal of something, from chunking
u basically like chunk everything except for like one thing
'''

import nltk

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

##train_text = state_union.raw("2005-GWBush.txt")
##sample_text = state_union.raw("2006-GWBush.txt")

f = open('QuestionSet_100.txt','rU')
train_text = f.read()

f = open('QuestionSet_200.txt','rU')
sample_text = f.read()

#train the Punkt tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

#actually tokenize
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r'''Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}'''
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()

    except Exception as e:
        print(str(e))


process_content()

'''
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when

'''
