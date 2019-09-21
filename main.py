import sys, os, json, shutil, time, traceback
from datetime import datetime

# Flask server stuff
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Data Science stuff: SKLearn / Pandas / NumPY
import pandas as pd
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import VectorizerMixin
import joblib
import numpy as np

# XAI stuff
import eli5
from eli5.lime import TextExplainer

# DB stuff
import psycopg2
from psycopg2.extras import DictCursor
try:
    connection_string = sys.argv[1]
except:
    print('need a DB connection string')
conn = psycopg2.connect(connection_string)
cursor = conn.cursor(cursor_factory=DictCursor)

# NLP stuff
DEMO = True
if DEMO:
    print("using localhost:9000 as word vector source")
    import requests
    #en_model = lambda word: json.loads(requests.get('http://localhost:9000/word/en?word=' + word).body)
    ar_model = lambda word: json.loads(requests.get('http://localhost:9000/word/ar?word=' + word).body)
else:
    print('loading text vectors')
    from gensim.models.keyedvectors import KeyedVectors
    ar_src = KeyedVectors.load_word2vec_format('wiki.ar.vec')
    def ar_model(word):
        if word not in ar_src:
            word = "the"
        return ar_src[word]
from nltk.tokenize import wordpunct_tokenize

# let's go
print('launching app')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

# pickled model
model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# These should be populated at training time
model_columns = None
clf = None
clfclasses = ['pos', 'neg']
text_type = True # False

# text tokenization and vectorization using FastText
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
                word_vec = phrase(word)
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
        df_ = pd.DataFrame(final_rows, columns=cols)
    else:
        df_ = df # removed include set

        for col, col_type in df_.dtypes.items():
            if col_type == 'O':
                categoricals.append(col)
            else:
                df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
    dependent_variable = list(df_.columns)[-1]
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

def new_model_id():
    nowtime = int(datetime.now().timestamp())
    cursor.execute("INSERT INTO models (created, updated) \
        VALUES (%s, %s)\
        RETURNING id", (nowtime, nowtime))
    conn.commit()
    row = cursor.fetchone()
    return row[0]

header_cache = {}
def get_headers(model_id):
    model_id = str(int(model_id))
    if model_id in header_cache:
        return header_cache[model_id]
    else:
        cursor.execute("SELECT * FROM rows_" + model_id + " LIMIT 1")
        response = list(cursor.fetchone().keys())
        header_cache[model_id] = response
        return response

def upload_csv_file(filename, table_code):
    fn = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return os.system('csvsql ' + fn + ' --db ' + connection_string + ' --tables rows_' + str(table_code) + ' --insert &')

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

    my_id = new_model_id()
    upload_csv_file(filename, my_id)

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    start = time.time()
    text_type = False
    fitme(x, y)

    return_message = {
        "status": "success",
        "train_time": (time.time() - start),
        "train_score": clf.score(x, y),
        "model_id": my_id
    }
    return jsonify(return_message)

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

    my_id = new_model_id()
    upload_csv_file(filename, my_id)

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    start = time.time()
    text_type = True
    fitme(x, y)

    return_message = {
        "status": "success",
        "train_time": (time.time() - start),
        "train_score": clf.score(x, y),
        "model_id": my_id
    }
    return jsonify(return_message)

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

@app.route('/training_data/<model_id>', methods=['GET'])
def tdata_html(model_id):
    with open('frontend/data-table.html') as content:
        return content.read()

@app.route('/training_data/headers/<model_id>', methods=['GET'])
def tdata_headers_api(model_id):
    return jsonify(get_headers(model_id))

@app.route('/training_data/api/<model_id>', methods=['GET'])
def tdata_api(model_id):
    model_id = str(int(model_id))
    cursor.execute('SELECT COUNT(*) FROM rows_' + model_id)
    count = cursor.fetchone()[0]

    sql_query = 'SELECT * FROM rows_' + model_id

    # ensures latest response comes back to table
    draw = int(request.args.get('draw'))

    # order
    order_col = request.args.get('order[0][column]')
    order_dir = request.args.get('order[0][dir]')
    if order_col is not None and order_dir in ['asc', 'desc']:
        order_col = int(order_col)
        headers = get_headers(model_id)
        sql_query += ' ORDER BY "' + headers[order_col].replace('"', '\\"') + '" ' + order_dir

    offset = str(int(request.args.get('start')))
    length = str(int(request.args.get('length')))
    sql_query += ' OFFSET ' + offset + ' LIMIT ' + length

    cursor.execute(sql_query)

    rows = []
    for rout in cursor.fetchall():
        row = []
        for col in rout.keys():
            row.append(rout[col])
        rows.append(row)

    return jsonify({
        "draw": draw,
        "recordsTotal": count,
        "recordsFiltered": count,
        "data": rows
    })

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
