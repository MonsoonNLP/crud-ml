import requests

url = 'http://localhost:8080'

# regular data post
files = { 'file': open('data/titanic.csv', 'rb') }
r = requests.post(url + '/train/create', files=files, data={})
print(r.content)

nextr = { 'file': open('data/titanic_extra.csv', 'rb') }
r2 = requests.post(url + '/train/insert', files=nextr, data={})
print(r2.content)

body = [
    {"Age": 85, "Sex": "male", "Embarked": "S"}
]
r3 = requests.post(url + '/predict', json=body)
print(r3.content)

# text data post
files = { 'file': open('data/nlp.csv', 'rb') }
r = requests.post(url + '/train_text/create', files=files, data={})
print(r.content)
