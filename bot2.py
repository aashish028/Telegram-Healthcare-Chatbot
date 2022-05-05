import re
from flask import Flask, request
import telegram
import random
import requests
import json
import numpy as np
import pickle
import nltk
import sys
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from flask import Response
lemmatizer = WordNetLemmatizer()

model = load_model("chatbot_model1.h5")
intents = json.loads(open("intent2_unedited - Copy.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

key = "5348611695:AAHg9m2NN-dnRMqVnT_KUzfKFT4G287bGuQ"
app = Flask(__name__)
def send_message(chat_id,text='Temp text'):
    url = f'https://api.telegram.org/bot{key}/sendMessage'
    payload = {'chat_id': chat_id, 'text': text} # Required parameters for sending a message is chat_id and text
    r = requests.post(url, json=payload)
    return r

def parse_telegramMessage(message):
    chat_id = message['message']['chat']['id']
    txt = message['message']['text']
    return chat_id, txt

@app.route("/",methods=["POST","GET"])
def index():
    if request.method == 'POST':
        try:
            msg = request.get_json()
            chat_id, txt = parse_telegramMessage(msg)
        
            if txt == 'START' or txt == 'start':
                send_message(chat_id, 'Welcome to our telegram bot, made to keep you healthy. Try using keywords instead of full sentences if the bot doesn\'t respond properly.')
                return Response('ok', status = 200)
    
            if txt== 'EXIT' or txt=='exit' or txt=='BYE' or txt=='bye' or txt=='Bye' or txt=='Goodbye' or txt=='GoodBye' or txt=='goodbye' or txt=='good bye':
                send_message(chat_id, 'Bye, have a nice day!')
                return Response('ok', status = 200)
        
            ints = predict_class(txt, model)
            res = getResponse(ints, intents)
            send_message(chat_id, res)
            return Response('ok', status = 200)
        except:
            send_message(chat_id, 'Sorry, I don\'t understand what you are saying. Try using keywords instead of a whole sentence.')
            return Response('ok', status = 200)
    else:
        return '<h1>Medical ChatBot</h1>'
    
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

    
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)
    
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

@app.route("/setwebhook/")
def setwebhook():
    url = "https://698f-2405-201-5c04-78f6-40c0-4458-d56e-4057.in.ngrok.io"
    s = requests.get("https://api.telegram.org/bot{}/setWebhook?url={}".format(key,url))
    if s:
        return "yes"
    else:
        return "fail"
    
    
if __name__ == "__main__":
    app.run(debug=True)