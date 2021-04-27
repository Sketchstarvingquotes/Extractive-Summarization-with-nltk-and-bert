# Extractive Text Summarization By using Nltk Python
-------------

### We will be using NLTK – the Natural Language Toolkit. 

### Extractive Summarization
In Extractive Summarization, we are identifying important phrases or sentences from the original text and extract only these phrases from the text, 
These extracted sentences would be the summary.

### required Installation
python 3.6 vertion or above
pip install nltk 
pip install heapq_max

**Note:-** commands used for installations as per OS above commands are based on windows)
* importing required libraries
* imprt re
* import nltk
* import heapq  
* from nltk.corpus import stopwords
* from nltk.tokenize import word_tokenize, sent_tokenize
* nltk.download('punkt')
* nltk.download('stopwords')

**Note:-** It's better to download nltk support libraries ontime of importing it such as nltk.download('punkt') And nltk.download('stopwords') both are support libraries of nltk.

### RegEx:-
In nltk extractive summarization we are using regex for text cleaning and removing unwanted text from article which we want to a summarize. 

### Corpus
A corpus is a large and structured set of machine-readable texts that have been produced in a natural communicative setting. Its plural is corpora. They can be derived in different ways like text that was originally electronic, transcripts of spoken language and optical character recognition, etc.
And also Corpus means a collection of text. It could be data sets of anything containing texts be it poems by a certain poet, bodies of work by a certain author, etc. In this case, we are going to use a data set of pre-determined stop words.

### Tokenizers
Tokenization is the process by which a large quantity of text is divided into smaller parts called tokens.
tokenizers – word, sentence, and regex tokenizer. In this project We will only use the word and sentence tokenizer this are the sub modules of  NLTK tokenize sentences.
**Tokenization of words**
We use the method word_tokenize() to split a sentence into words. The output of word tokenization can be converted to Data Frame for better text understanding in machine learning applications.
**Tokenization of Sentences**
Sub-module available for the above is sent_tokenize it's used for when we need to count average words per sentence, For accomplishing such a task, we need both NLTK sentence tokenizer as well as NLTK word tokenizer to calculate the ratio, 
Such output serves as an important feature for machine training as the answer would be numeric.
# Implimentation Steps

## Preprocessing
Preprocessing is the first step which are used to remove references from Our text file like Wikipedia article or refarance papers, references are enclosed in square brackets. The following script removes the square brackets and replaces the resulting multiple spaces by a single space. Take a look at the script below.

<pre>
### Removing Square Brackets and Extra Spaces
mystring = re.sub(r'\[[0-9]*\]', ' ', mystring)
mystring = re.sub(r'\s+', ' ', mystring)
</pre>

The mystring object contains text without brackets and without extra spaces However, we do not want to remove anything else from the our text file stored in mystring since this is the original text. We will not remove other numbers, punctuation marks and special characters from this text since we will use this text to create summaries and weighted word frequencies will be replaced in this text data.
For further step like clean the text and calculate weighted frequences, we will create another object. 

<pre>
# Here we are Removing special characters and digits
clear_textfile = re.sub('[^a-zA-Z]', ' ', mystring)
clear_textfile = re.sub(r'\s+', ' ', clear_textfile)
</pre>
 
At this point we have two objects one is mystring which contains our original text and clear_textfile which contains the formatted text or cleaned text. We will use clear_textfile to create weighted frequency histograms for the words and will replace these weighted frequencies with the words in the mystring object.

## Converting Text To Sentences 
At this point we have preprocessed the data. Next, we need to tokenize the text into sentences. We will use the mystring object for tokenizing the text to sentence since it contains full stops. The clear_textfile does not contain any punctuation and therefore cannot be converted into sentences using the full stop as a parameter.
following script performs sentence tokenization:
<pre>
sentence_list = nltk.sent_tokenize(mystring)
</pre>

## Need to Find Weighted Frequency of Occurrence
To find the frequency of occurrence of each word, we use the clear_textfile variable. We used this variable to find the frequency of occurrence since it doesn't contain punctuation, digits, or other special characters. Take a look at the following implimentation:

<pre>
stopwords = nltk.corpus.stopwords.words('english')
word_frequencies = {}
for word in nltk.word_tokenize(clear_textfile):
	if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
</pre>

In the above code implimentation, we first store all the English stop words from the nltk library into a stopwords variable. Next, we loop through all the sentences and then corresponding words to first check if they are stop words. If not, we proceed to check whether the words exist in word_frequency dictionary i.e. word_frequencies, or not. If the word is encountered for the first time, it is added to the dictionary as a key and its value is set to 1. Otherwise, if the word previously exists in the dictionary, its value is simply updated by 1.
Finally, to find the weighted frequency, we can simply divide the number of occurances of all the words by the frequency of the most occurring word, as shown below:
<pre>
max_freq = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/max_freq)
</pre>
## Calculating Sentence Scores
We have now calculated the weighted frequencies for all the words. Now is the time to calculate the scores for each sentence by adding weighted frequencies of the words that occur in that particular sentence. The following script calculates sentence scores:
<pre>
finalsentence = {}
for sent in sentence_list:
for word in nltk.word_tokenize(sent.lower()):
  if word in word_frequencies.keys():
    if len(sent.split(' ')) < 25:
      if sent not in finalsentence.keys():
        finalsentence[sent] = word_frequencies[word]
      else:
        inalsentence[sent] += word_frequencies[word]
</pre>

In the above code or script, we first create an empty finalsentence dictionary. The keys of this dictionary will be the sentences themselves and the values will be the corresponding scores of the sentences. Next, we loop through each sentence in the sentence_list and tokenize the sentence into words.
We then check if the word exists in the word_frequencies dictionary. This check is performed since we created the sentence_list list from the mystring object; on the other hand, the word frequencies were calculated using the clear_textfile object, which doesn't contain any stop words, numbers, etc.
We do not want very long sentences in the summary, therefore, we calculate the score for only sentences with less than 25 words (although you can increase or decrease this value as per requirment). Next, we check whether the sentence exists in the sentence_scores dictionary or not. If the sentence doesn't exist, we add it to the sentence_scores dictionary as a key and assign it the weighted frequency of the first word in the sentence, as its value. On the contrary, if the sentence exists in the dictionary, we simply add the weighted frequency of the word to the existing value.

## Getting the Summary(Extracting summary)
Now we have the  finalsentence dictionary that contains sentences with their corresponding score. To summarize the text article, we can take top N sentences with the highest scores. The following script retrieves top 8 sentences and prints them on the screen.
<pre>
import heapq
sentsummary = heapq.nlargest(8, finalsentence, key=finalsentence.get)
summary = ' '.join(sentsummary)
print('Summery:-',summary)
</pre>
In the script above, we use the heapq library and call its nlargest function to retrieve the top 8 sentences with the highest scores and better summary result.
