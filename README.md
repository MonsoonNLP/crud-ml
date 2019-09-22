# CRUD ML

This repo is based directly on Amir Ziai's <a href="https://github.com/amirziai/sklearnflask">SKLearnFlask</a> ML server, which serves predictions from a scikit-learn model.

The new features:
- parses POST-ed CSVs
- incremental learning via SciKit-Learn
- explainable AI via ELI5
- SQL database of submitted data rows
- adjustable final layer for word weights

### Customizing

The database is not required for training or prediction, but is useful here to identify models,
show training data, and add word weights. In ```main.py``` you can set ```DATABASE = False``` and the next two
lines for parameters.

For text data / NLP, I chose <a href="https://fasttext.cc/docs/en/pretrained-vectors.html">FastText</a>
 as the pre-trained word embeddings source for its many languages. You could use any Word2Vec-compatible
source. With some additional coding, you could replace the Vectorizer class with GPT-2 with HuggingFace's
<a href="https://github.com/huggingface/pytorch-transformers">PyTorch-Transformers</a>
 (
  <a href="https://github.com/mapmeld/tweet_classifier_plus_eli5/blob/master/gpt2_sklearn.py">see example</a>
), or SciKit-Learn's own TfidfVectorizer (
  <a href="https://github.com/mapmeld/tweet_classifier_plus_eli5/blob/master/basic_sklearn.py">see example</a>
).

I chose ```SGDClassifier``` as the SciKit-Learn model, as only this, ```Perceptron```, and ```PassiveAggressiveClassifier``` support both incremental learning and ELI5 native integration. If you use only text data, and use vectorizers outside of SciKit-Learn (including the current FastText/word2vec setup), the vectorizer will break ELI5 integration and need to use the LIME algorithm. If you are in this situation, you are free to switch to any
<a href="https://scikit-learn.org/stable/modules/computing.html#incremental-learning">incremental learning classifier</a>.

### Dependencies
- scikit-learn
- Flask
- pandas
- numpy
- joblib
- requests
- csvkit
- gensim
- nltk
- eli5
- psycopg2

### Creating database
You need PostgreSQL.

```bash
initdb crud-db
postgres -D crud-db
createdb crud-db
python3 seed-db.py postgresql:///crud-db
```

### Running API
```
python3 main.py postgresql:///crud-db
```

# Endpoints
### /predict/:model_id (POST)
Returns an array of predictions given a JSON object representing independent variables. Here's a sample input:
```
[
    {"Age": 85, "Sex": "male", "Embarked": "S"},
    {"Age": 24, "Sex": "female", "Embarked": "C"},
    {"Age": 3, "Sex": "male", "Embarked": "C"},
    {"Age": 21, "Sex": "male", "Embarked": "S"}
]
```

and sample output:
```
{"prediction": [0, 1, 1, 0], "explanations": [{...}, {}, {}, {}]}
```

### /train/create (POST)
Trains a classifier model with numeric or categorical values from a CSV file.

#### /train/insert/:model_id (POST)
Adds more rows to incrementally train the classifier model of numeric and categorical values.

### /train_text/create (POST)
Use FastText to tokenize and vectorize the text of your first column, and expect other columns
to be numeric (should add categorical later).

#### /train_text/insert/:model_id (POST)
Adds more rows to incrementally train the text classifier model.

### /delete/:model_id (GET)
Deletes the trained model and its training data.

### /training_data/:model_id (GET)
HTML / JS table of training data

### /predict_hub/:model_id (GET)
HTML / JS UI for predictions (text models only)
