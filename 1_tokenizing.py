#import nltk;


# corpora - body of text
# lexicon - words and their means

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

f = open('QuestionSet.txt','rU')
text = f.read()
text1 = text.split()
abstracts = nltk.Text(text1)

print(sent_tokenize(EXAMPLE_TEXT))
print(word_tokenize(text))

for i in sent_tokenize(EXAMPLE_TEXT):
    print (i)
