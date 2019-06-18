
# coding: utf-8

# In[12]:


import spacy
from spacy.lang.en.examples import sentences
nlp = spacy.load('en_core_web_lg')


# In[22]:




nlp = spacy.load('en_core_web_lg')
doc = nlp(u'''Andrew Yan-Tak Ng is a Chinese American computer scientist.
He is the former chief scientist at Baidu, where he led the company's
Artificial Intelligence Group. He is an adjunct professor (formerly 
associate professor) at Stanford University. Ng is also the co-founder
and chairman at Coursera, an online education platform. Andrew was born
in the UK in 1976. His parents were both from Hong Kong.''')

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

