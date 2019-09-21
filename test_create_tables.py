import requests

url = 'http://localhost:8080'

# regular data post
files = { 'file': open('data/titanic.csv', 'rb') }
r = requests.post(url + '/train/create', files=files, data={})
print(r.content)
