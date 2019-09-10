import sys, os, json, shutil, time, traceback

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import pandas as pd
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import VectorizerMixin

import joblib
import numpy as np

import eli5
from eli5.lime import TextExplainer

print('loading text vectors')
from nltk.tokenize import wordpunct_tokenize

en_model = lambda word: json.loads(requests.get('http://localhost:9000/word/en?word=' + word).body)
ar_model = lambda word: json.loads(requests.get('http://localhost:9000/word/ar?word=' + word).body)

print('launching app')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

# inputs
training_data = 'data/titanic.csv'
include = ['Age', 'Sex', 'Embarked', 'Survived']
dependent_variable = include[-1]

model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# These will be populated at training time
model_columns = None
clf = None
clfclasses = ['pos', 'neg']
text_type = True # False

class V(VectorizerMixin):
    def fit (self, X, y=None):
        return self

    def transform (self, X):
        rows = []
        for item in X:
            if 'text' in item:
                row = []
                text_src = item['text']
                #del item['text']
                words = wordpunct_tokenize(text_src)
                sentence_vecs = []
                for w in range(0, len(words)):
                    word = words[w]
                    word_vec = ar_model(word)
                    for v in range(0, len(word_vec)):
                        if w == 0:
                            item['avg_' + str(v)] = 0.0
                            #item['max_' + str(v)] = word_vec[v]
                            #item['min_' + str(v)] = word_vec[v]
                        item['avg_' + str(v)] += float(word_vec[v]) / float(len(words))
                        #item['max_' + str(v)] = max(word_vec[v], item['max_' + str(v)])
                        #item['min_' + str(v)] = min(word_vec[v], item['min_' + str(v)])
            rows.append(item)

        query = pd.get_dummies(pd.DataFrame(rows))
        query = query.reindex(columns=model_columns, fill_value=0)
        return query

vectorizer = V()

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:

            # https://github.com/amirziai/sklearnflask/issues/3
            # Thanks to @lorenzori

            explainers = []
            if text_type:
                pipe = make_pipeline(vectorizer, clf)
                prediction = pipe.predict(request.json)

                for post in request.json:
                    te = TextExplainer(random_state=42, n_samples=500)
                    te.fit(post['text'], pipe.predict_proba)
                    made = te.explain_prediction(target_names=['pos', 'neg'])
                    explanation = made.targets[0].feature_weights
                    op_exp = {'pos': [], 'neg': []}
                    for feature in explanation.pos:
                        op_exp['pos'].append([feature.feature, feature.weight])
                    for feature in explanation.neg:
                        op_exp['neg'].append([feature.feature, feature.weight])
                    explainers.append(op_exp)
            else:
                rows = request.json
                query = pd.get_dummies(pd.DataFrame(rows))
                query = query.reindex(columns=model_columns, fill_value=0)
                prediction = clf.predict(query)
                for index, row in query.iterrows():
                    explanation = eli5.explain_prediction(clf, row).targets[0].feature_weights
                    op_exp = {'pos': [], 'neg': []}
                    for feature in explanation.pos:
                        op_exp['pos'].append([feature.feature, feature.weight])
                    for feature in explanation.neg:
                        op_exp['neg'].append([feature.feature, feature.weight])
                    explainers.append(op_exp)

            # Converting to int from int64
            return jsonify({
                "predictions": list(map(str, prediction)),
                "explanations": explainers
            })

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(request):
    if 'file' not in request.files:
        raise Exception('no file included in POST')
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename
    else:
        raise Exception('need CSV file for upload')

def process_csv(filename, vectorize_text=False):
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    categoricals = []  # going to one-hot encode categorical variables

    if vectorize_text:
        # first column is text
        # last column is dependent variable for classifier
        final_rows = []
        cols = []
        for index, srs in df.iterrows():
            row = list(srs)
            words = wordpunct_tokenize(row[0])
            sentence_vecs = []
            # print(row[1:])
            for w in range(0, len(words)):
                word = words[w]
                word_vec = ar_model(word])
                for v in range(0, len(word_vec)):
                    if w == 0:
                        sentence_vecs.append(0.0)
                        #sentence_vecs.append(word_vec[v])
                        #sentence_vecs.append(word_vec[v])
                        if index == 0:
                            cols += ['avg_' + str(v)] #, 'max_' + str(v), 'min_' + str(v)]
                    sentence_vecs[v * 1] += float(word_vec[v]) / float(len(words))
                    #sentence_vecs[v * 3 + 1] = max(word_vec[v], sentence_vecs[v * 3 + 1])
                    #sentence_vecs[v * 3 + 2] = min(word_vec[v], sentence_vecs[v * 3 + 2])
            sentence_vecs += row[1:]
            if index == 0:
                cols += list(df.columns)[1:]
            final_rows.append(sentence_vecs)
        dependent_variable = cols[-1]
        df_ = pd.DataFrame(final_rows, columns=cols)
    else:
        # include array is columns of variables used in decision
        # last column is dependent variable for classifier
        df_ = df[include]
        dependent_variable = include[-1]

        for col, col_type in df_.dtypes.items():
            if col_type == 'O':
                categoricals.append(col)
            else:
                df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]
    return (x, y)

def fitme(x, y):
    global clf, clfclasses
    clf = SGDClassifier(loss='log')
    if len(clfclasses) == 0:
        clfclasses = np.unique(y)
    clf.partial_fit(x, y, classes=clfclasses)
    joblib.dump(clf, model_file_name)
    return clf

@app.route('/train/create', methods=['POST'])
def create_train():
    try:
        filename = validate_file(request)
    except Exception as e:
        print(e)
        return str(e)

    global clfclasses
    clfclasses = []
    (x, y) = process_csv(filename)

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    start = time.time()
    text_type = False
    fitme(x, y)

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % clf.score(x, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2)
    return return_message

@app.route('/train_text/create', methods=['POST'])
def create_text():
    try:
        filename = validate_file(request)
    except Exception as e:
        print(e)
        return str(e)

    global clfclasses
    clfclasses = []
    (x, y) = process_csv(filename, vectorize_text=True)

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    start = time.time()
    text_type = True
    fitme(x, y)

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % clf.score(x, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2)
    return return_message

@app.route('/train/insert', methods=['POST'])
def insert_train():
    if clf:
        try:
            filename = validate_file(request)
        except Exception as e:
            print(e)
            return str(e)

        text_type = False
        (x, y) = process_csv(filename)
        fitme(x, y)
        return 'Success\nModel training score: %s' % clf.score(x, y)
    else:
        print('train first')
        return 'no model here'

@app.route('/train_text/insert', methods=['POST'])
def insert_text():
    if clf:
        try:
            filename = validate_file(request)
        except Exception as e:
            print(e)
            return str(e)

        text_type = True
        (x, y) = process_csv(filename, vectorize_text=True)
        fitme(x, y)
        return 'Success\nModel training score: %s' % clf.score(x, y)
    else:
        print('train first')
        return 'no model here'

@app.route('/train/delete', methods=['GET'])
def delete_train():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print(e)
        return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 8080

    try:
        clf = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=True)
