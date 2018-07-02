import nltk

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
#PunktSentenceTokenizer is an unsupervised machine learning sentence tokenizer. It comes pre trained, but u can train it if u want.

#One is a State of the Union address from 2005, and the other is from 2006 from past President George W. Bush.
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

#train the Punkt tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

#actually tokenize
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)        
            namedEnt = nltk.ne_chunk(tagged)  
            namedEnt.draw()



    except Exception as e:
        print(str(e))


process_content()

'''["PRESIDENT GEORGE W. BUSH'S ADDRESS BEFORE A JOINT SESSION OF THE CONGRESS ON THE STATE OF THE UNION\n \nJanuary 31, 2006\n\nTHE PRESIDENT: Thank you all.",
'Mr. Speaker, Vice President Cheney, members of Congress, members of the Supreme Court and diplomatic corps, distinguished guests, and fellow citizens:
Today our nation lost a beloved, graceful, courageous woman who called America to its founding ideals and carried on a noble dream.',
'Tonight we are comforted by the hope of a glad reunion with the husband who was taken so long ago, and we are grateful for the good life of Coretta Scott King.',
'(Applause.)',
'President George W. Bush reacts to applause during his State of the Union Address at the Capitol, Tuesday, Jan.']
'''
