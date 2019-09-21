import requests
import time

url = 'http://localhost:8080'

# regular data post
files = { 'file': open('data/titanic.csv', 'rb') }
r = requests.post(url + '/train/create', files=files, data={})
print(r.content)
model_id = r.json()["model_id"]

print('sleep')
time.sleep(3)
print('wakeup')

# append
nextr = { 'file': open('data/titanic_extra.csv', 'rb') }
r2 = requests.post(url + '/train/insert/' + str(model_id), files=nextr, data={})
print(r2.content)

print('sleep')
time.sleep(3)
print('wakeup')

# delete
r3 = requests.get(url + '/delete/' + str(model_id))
print(r3.content)
