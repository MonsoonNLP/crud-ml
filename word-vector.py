import json
from sys import argv

from flask import Flask, request, jsonify

from gensim.models.keyedvectors import KeyedVectors
try:
    ar_model = KeyedVectors.load_word2vec_format('wiki.ar.vec')
    en_model = KeyedVectors.load_word2vec_format('wiki.en.vec')
except:
    print('Arabic word vectors not in same directory')

app = Flask(__name__)

@app.route('/word/en')
def word():
    word = request.args.get('word')
    if word in en_model:
        return en_model[word]
    else:
        return en_model['the']

@app.route('/word/ar')
def word():
    word = request.args.get('word')
    if word in ar_model:
        return ar_model[word]
    else:
        return ar_model['the']

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 9000
    app.run(host='0.0.0.0', port=port, debug=True)
