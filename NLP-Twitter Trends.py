#!/usr/bin/env python
# coding: utf-8

# # Important packages

# In[1]:


#import important packages
import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
pd.set_option('display.max_colwidth',100)


# In[2]:


#Check working directory
os.getcwd()


# In[3]:


# set your working directory
os.chdir(r'C:\Users\admin\Downloads\Course_Content\Employable_NLP_Content')


# In[4]:


#import the twitter data
tweet = pd.read_excel('tweets.xlsx',0)


# In[5]:


# Check the dimension of data
print("Number of rows in data =",tweet.shape[0])
print("Number of columns in data =",tweet.shape[1])
print("\n")
print("**Sample data:**")
print(tweet.head())


# 

# In[6]:


import nltk


# In[7]:


# download all the data in the NLTK downloader pop up window
nltk.download()


# In[8]:


import nltk.corpus


# In[9]:


#Different samples, words, movie review, so many files
print(os.listdir(nltk.data.find("corpora")))


# In[10]:


# viewing all the functions, attributes, and methods within nltk package
dir(nltk)


# # Tokenization

# In[11]:


# Tokenize the data
#A sentence or data can be split into words using the method word_tokenize():
from nltk.tokenize import sent_tokenize, word_tokenize
 
data = "All work and no play makes jack a dull boy. All work and no play"
print(word_tokenize(data))


# In[12]:


#Tokenizing sentences
from nltk.tokenize import sent_tokenize, word_tokenize
 
data = "All work and no play makes jack a dull boy. All work and no play"
print(sent_tokenize(data))


# In[13]:


#If you wish to you can store the words and sentences in arrays:
from nltk.tokenize import sent_tokenize, word_tokenize
 
data = "All work and no play makes jack a dull boy. All work and no play"
 
phrases = sent_tokenize(data)
words = word_tokenize(data)
 
print(phrases)
print(' ')
print(words)


# # N Gram

# In[14]:


import nltk
from nltk import ngrams


# In[15]:


sentence = 'this is a foo bar sentences and i want to ngramize it'


# In[16]:


n = 2
bigrams = ngrams(sentence.split(), n)


# In[17]:


for grams in bigrams:
  print (grams)


# # Frequency Distribution

# In[18]:


len(words)


# In[19]:


from nltk.probability import FreqDist
fdist = FreqDist()


# In[20]:


for word in words:
    fdist[word.lower()]+=1
fdist


# In[21]:


fdist.most_common(50)


# # StopWords

# In[22]:


# an example to view the stopwords in nltk library
from nltk.corpus import stopwords

# view all the stopwords in english langauge with an incremenet of 25 up to 500 words
stopwords.words("english")[0:500:25]


# In[23]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)
print(word_tokens)


# In[24]:


filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)


# In[25]:


filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence


# # Remove Punctuation

# In[26]:


import string
from string import punctuation
string.punctuation


# In[27]:


def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
text = "Hello!+ how <are> you\\ doing|?"
print (strip_punctuation(text))


# # Remove Tags

# In[28]:


import re
text = """<head><body>hello world!</body></head>"""
cleaned_text = re.sub('<[^<]+?>','', text)
print (cleaned_text)


# # Removing Numbers

# In[29]:


text = "There was 200 people standing right next to me at 2pm."
output = ''.join(c for c in text if not c.isdigit())
print(output)


# # Regular Expression

# <table ><tr><th>Character</th><th>Description</th><th>Example Pattern Code</th><th >Exammple Match</th></tr>
# 
# <tr ><td><span >\d</span></td><td>A digit</td><td>file_\d\d</td><td>file_25</td></tr>
# 
# <tr ><td><span >\w</span></td><td>Alphanumeric</td><td>\w-\w\w\w</td><td>A-b_1</td></tr>
# 
# 
# 
# <tr ><td><span >\s</span></td><td>White space</td><td>a\sb\sc</td><td>a b c</td></tr>
# 
# 
# 
# <tr ><td><span >\D</span></td><td>A non digit</td><td>\D\D\D</td><td>ABC</td></tr>
# 
# <tr ><td><span >\W</span></td><td>Non-alphanumeric</td><td>\W\W\W\W\W</td><td>*-+=)</td></tr>
# 
# <tr ><td><span >\S</span></td><td>Non-whitespace</td><td>\S\S\S\S</td><td>Yoyo</td></tr></table>

# In[30]:


# importing regex package
import re

text = 'Emploab11, study is fun'


# In[31]:


#"^"   : This expression matches the start of a string
# w   : will give the first character of the first letter
#"w+" : This expression matches the alphanumeric character in the string
#"s"  : This expression is used for creating a space in the string


# In[32]:


print(re.findall('^\w', text)) 
print(re.findall('^\w+', text))
print(re.split('\s', text))


# In[33]:


# sample sentences 
re_test_messy = 'This      is a made up     string to test 2    different regex methods'
re_test_messy1 = 'This-is-a-made/up.string*to>>>>test----2""""""different~regex-methods'


# In[34]:


# \s means single whitespace
re.split('\s', re_test_messy)


# In[35]:


# \s+ means multiple (one or more) whitespaces  
re.split('\s+', re_test_messy)


# In[36]:


# \s+ means multiple (one or more) whitespaces  
re.split('\s+', re_test_messy1) # will not work as there are no whitespace in the string


# In[37]:


# \W+ means multiple (one or more) non-word characters (Method 1: splitting charachters by searching the words)
re.split('\W+', re_test_messy1)


# In[38]:


# \S+ means multiple (one or more) non-whitespace words (Method 2: ignoring charachters which split the words)
re.findall('\S+', re_test_messy)


# In[39]:


# \w+ means multiple (one or more) non-word characters (Method 2: ignoring charachters which split the words)
re.findall('\w+', re_test_messy1)


# ### Starts With and Ends With
# 
# We can use the **^** to signal starts with, and the **$** to signal ends with:

# In[40]:


# Ends with a number
re.findall(r'\d$','This ends with a number 2')


# In[41]:


# Starts with a number
re.findall(r'^\d','1 is the loneliest number.')


# # Repalcing a specific string

# In[42]:


pep8_test = 'I try to follow PEP8 guidelines'
pep7_test = 'I try to follow PEP7 guidelines'
peep8_test = 'I try to follow PEEP8 guidelines'


# In[43]:


# using findall to find words which contains characters from a-z 
re.findall('[a-z]+', pep8_test) # regex is case sensitive


# In[44]:


# using findall to find words which contains characters from A-Z  
re.findall('[A-Z]+', pep8_test) # regex is case sensitive


# In[45]:


# using findall get PEP8 word. [A-Z0-9] means finding charachters containing upper case letters OR numbers
re.findall('[A-Z0-9]+', pep8_test)


# In[46]:


# using findall to get only the PEP8 word. [A-Z][0-9] means finding charachters containing upper case letters AND numbers
re.findall('[A-Z]+[0-9]+', pep8_test)


# In[47]:


# using findall to get only the PEP8 word. [A-Z][0-9] means finding charachters containing upper case letters AND numbers
re.findall('[A-Z]+[0-9]+', pep7_test)


# In[48]:


# using findall to get only the PEP8 word. [A-Z][0-9] means finding charachters containing upper case letters AND numbers
re.findall('[A-Z]+[0-9]+', peep8_test)


# In[49]:


# replacing PEP7 and PEEP8 by PEP using the above regex pattern in pep7_test and peep8_test using sub function
re.sub('[A-Z]+[0-9]+', 'PEP8 Python style', pep8_test)


# In[50]:


re.sub('[A-Z]+[0-9]+', 'PEP8 Python style', pep7_test)


# In[51]:


re.sub('[A-Z]+[0-9]+', 'PEP8 Python style', peep8_test)


# In[52]:


#re.match()- The match function is used to match the RE pattern to string with optional flags
list = ['Employable platform','edwisor platform','Xy','XYZ']

for element in list:
    z =re.match('X\w+', element)
    if z:
        print(z)


# In[53]:


#re.search()- to search for a pattern in a tex
patterns =['data science','java']
text= 'Employable will provide you platform to study data science'
for i in patterns:
    print(' Looking for "%s" in "%s"  ->' %(i, text), end='')
    if re.search(i,text):
        print(' Match found')
    else:
        print(' Match not found')


# In[54]:


#re.finadall()- Re.findall() module is used when you want to iterate over the lines of the file
lis = 'abc@gmail.com , XYZmail.com ,red@yahoomail.com ,WAR@gmailcom'
emails = re.findall(r'[\w\.-]+@[\w\.-]+', lis)

for email in emails:
    print(email)


# # Stemming

# In[55]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()


# In[56]:


ps.stem('having')


# In[57]:


example_words = ["python","pythoner","pythoning","pythoned","pythonly"]


# In[58]:


for w in example_words:
    print(ps.stem(w))


# In[59]:


#Porter stemming
from nltk.stem import PorterStemmer
ps = nltk.PorterStemmer()


# In[60]:


text = 'This is a Demoes Texting for NLP using NLTK. Full forms of NLTK is Natural Language Toolkit'

word_tokens = nltk.word_tokenize(text)
stemmed_word = [ps.stem(word) for word in word_tokens]
print (stemmed_word)


# In[61]:


# Snowball stemming
from nltk.stem import SnowballStemmer
sbs = SnowballStemmer("english") #need to specify language


# In[62]:


text = 'This is a Demoes Texting for NLP using NLTK. Full forms of NLTK is Natural Language Toolkit'

word_tokens = nltk.word_tokenize(text)
stemmed_word = [sbs.stem(word) for word in word_tokens]
print (stemmed_word)


# In[63]:


# Lancaster Stemming
from nltk.stem import LancasterStemmer
lst = LancasterStemmer()


# In[64]:


text = 'This is a Demoes Texting for NLP using NLTK. Full forms of NLTK is Natural Language Toolkit'

word_tokens = nltk.word_tokenize(text)
stemmed_word = [lst.stem(word) for word in word_tokens]
print (stemmed_word)


# In[65]:


dir(nltk.stem)


# # Lemmatization

# In[66]:


#Wordnet Lemmatizer
#Spacy Lemmatizer
#TextBlob
#CLiPS Pattern
#Stanford CoreNLP
#Gensim Lemmatizer
#TreeTagger


# In[67]:


sentence = """Following mice attacks, caring farmers were marching to Delhi for better living conditions. 
Delhi police on Tuesday fired water cannons and teargas shells at protesting farmers as they tried to 
break barricades with their cars, automobiles and tractors."""


# In[68]:


# Tokenize: Split the sentence into words

word_list = nltk.word_tokenize(sentence)
print(word_list)


# In[69]:


# NLTK
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_output = " ".join([lemmatizer.lemmatize(w) for w in word_list if w not in string.punctuation])
print(lemmatized_output)


# In[70]:


#with POS
print(lemmatizer.lemmatize("stripes", 'v')) 

print(lemmatizer.lemmatize("stripes", 'n')) 


# In[71]:


#Wordnet Lemmatizer with appropriate POS tag
print(nltk.pos_tag(nltk.word_tokenize(sentence)))


# In[72]:


# # Download Wordnet through NLTK in python console:
import nltk
nltk.download('wordnet')


# In[73]:


# Define the sentence to be lemmatized
sentence = 'This is a Demoes Texting for NLP using NLTK Full forms of NLTK is Natural Language Toolkit'

# Tokenize: Split the sentence into words
word_list = nltk.word_tokenize(sentence)
print(word_list)


# In[74]:


# 1. Init the Wordnet Lemmatizer
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()
# Lemmatize list of words and join
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
print(lemmatized_output)


# ## Count Vectorizer
# ####  https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# In[75]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[76]:


corpus =["The car is driven on the road",
         "The truck is driven on the highway"]


# In[77]:


count_vector=cv.fit_transform(corpus)
count_vector.shape


# In[78]:


x= cv.fit_transform(corpus)
# show resulting vocabulary; the numbers are not counts, they are the position in the sparse vector.
print(cv)
print(cv.vocabulary_)
print(cv.get_feature_names())


# In[79]:


print(x.shape)
print(x.toarray())


# In[80]:


# adding stop words
cv = CountVectorizer(corpus,stop_words=["I","to","and"])
count_vector=cv.fit_transform(corpus)
print(count_vector.shape)
print(count_vector.shape)
print(count_vector.toarray())


# ## TF-IDF
# #### https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

# In[81]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()


# In[82]:


x= tfidf_vect.fit(corpus)
print(x.vocabulary_)
print(tfidf_vect.get_feature_names())

x=tfidf_vect.transform(corpus)


# In[83]:


print(x.shape)
print(x)
print(x.toarray())


# In[84]:


df = pd.DataFrame(x.toarray(),columns=tfidf_vect.get_feature_names())
df


# # Data cleaning

# In[85]:


data = pd.read_excel('tweets.xlsx',0)
data.columns =['body_text','label']
data.head()


# In[86]:


def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))


# In[87]:


import re

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

data['body_text_tokenized'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower()))


# In[88]:


import nltk

stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x))


# In[89]:


data = pd.read_excel('tweets.xlsx',0)
data.columns =['body_text','label']


# In[90]:


import nltk

wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')


# In[91]:


def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text

data['body_text_nostop'] = data['body_text'].apply(lambda x: clean_text(x.lower()))

data.head()


# In[92]:


def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

data['body_text_lemmatized'] = data['body_text_nostop'].apply(lambda x: lemmatizing(x))

data.head(10)


# # Word Cloud 

# In[93]:


# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt


# In[94]:


data = pd.read_excel('tweets.xlsx',0)
data.columns =['body_text','label']
data.head()


# In[95]:


# Start with one review:
text = data.body_text[0]
print(text)
# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)


# In[96]:


# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[97]:


text = " ".join(review for review in data.body_text)
"".join([word for word in text if word not in string.punctuation])
print ("There are {} words in the combination of all review.".format(len(text)))


# In[98]:


# Create stopword list:
stopwords = set(STOPWORDS)

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[99]:


# Save the image in the img folder:
wordcloud.to_file("first_review.png")


# In[100]:


import pandas as pd
import re
import string
import nltk
pd.set_option('display.max_colwidth', 100)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_excel('tweets.xlsx',0)
data.columns =['body_text','label']
data.head()


# In[101]:


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


# In[102]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())


# In[103]:


df = pd.DataFrame(X_tfidf.toarray(),columns=tfidf_vect.get_feature_names())
df.head()


# In[ ]:





# In[ ]:




