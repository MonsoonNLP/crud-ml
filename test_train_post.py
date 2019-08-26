import requests

url = 'http://localhost:8080/train'

files = { 'file': open('data/titanic.csv', 'rb') }
r = requests.post(url + '/create', files=files, data={})
print(r.content)

nextr = { 'file': open('data/titanic_extra.csv', 'rb') }
r = requests.post(url + '/insert', files=nextr, data={})
print(r.content)
