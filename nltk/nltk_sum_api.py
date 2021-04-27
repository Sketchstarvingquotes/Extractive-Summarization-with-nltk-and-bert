from flask import Flask,request,jsonify
import re
import heapq
import nltk  
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
import json

app = Flask(__name__)
@app.route('/api/v1/summerize', methods=["POST"])

def Text_sum():
    json_data = request.json
    # replace key which you want to summerize
    datalist = list(map(lambda x: x["text"], json_data))
    str = " "
    mystring = str.join(datalist)
    mystring = re.sub(r'\[[0-9]*\]', ' ', mystring)
    mystring = re.sub(r'\s+', ' ', mystring)
    clear_textfile = re.sub('[^a-zA-Z]', ' ', mystring)
    clear_textfile = re.sub(r'\s+', ' ', clear_textfile)
    sentence_list = nltk.sent_tokenize(mystring)
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(clear_textfile):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    max_freq = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/max_freq)

    finalsentence = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 25:
                    if sent not in finalsentence.keys():
                        finalsentence[sent] = word_frequencies[word]
                    else:
                        finalsentence[sent] += word_frequencies[word]
    sentsummary = heapq.nlargest(8, finalsentence, key=finalsentence.get)
    summary = ' '.join(sentsummary)
    return jsonify('Summery:-',summary)


app.run(host="0.0.0.0", port=3003)
