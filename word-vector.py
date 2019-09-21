import json
from sys import argv

from flask import Flask, request, jsonify

from gensim.models.keyedvectors import KeyedVectors
try:
    ar_model = KeyedVectors.load_word2vec_format('wiki.ar.vec')
    en_model = KeyedVectors.load_word2vec_format('wiki.en.vec')
except:
    ar_model = { 'the': [1,2,3] }
    en_model = { 'the': [1,2,3] }
    print('Arabic and/or English word vectors not in same directory')

app = Flask(__name__)

@app.route('/word/en')
def en_word():
    word = request.args.get('word')
    if word not in en_model:
        word = 'the'
    return jsonify(en_model[word])

@app.route('/word/ar')
def ar_word():
    word = request.args.get('word')
    if word not in ar_model:
        word = 'the'
    return jsonify(ar_model[word])

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 9000
    app.run(host='0.0.0.0', port=port, debug=True)
