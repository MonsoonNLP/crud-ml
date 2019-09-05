import csv
from random import shuffle
import requests

url = 'http://178.62.232.92:8080'
all = []

with open('data/Arabic_tweets_positive_20190413.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        all.append([row[1], row[0]])
with open('data/Arabic_tweets_negative_20190413.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        all.append([row[1], row[0]])
shuffle(all)

for fragment in range(0, round(len(all) / 1500 + 0.5)):
    with open('data/combined_arabic_' + str(fragment) + '.csv', mode='w') as final:
        wr = csv.writer(final, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['text', 'category'])
        for row in all[ fragment * 1500 : (fragment + 1) * 1500 ]:
            wr.writerow(row)

print('creating initial model')
files = { 'file': open('data/combined_arabic_0.csv', 'r') }
r = requests.post(url + '/train_text/create', files=files, data={}, verify=False)
print(r.content)

for fragment in range(1, round(len(all) / 1500 + 0.5)):
    print('uploading ' + str(fragment))
    files = { 'file': open('data/combined_arabic_' + str(fragment) + '.csv', 'r') }
    r = requests.post(url + '/train_text/insert', files=files, data={}, verify=False)
    print(r.content)

predict = 'إذا مزاجك تنهي الشهر بويك إند مرعب 👻 - Clown قناع عيد ميلاد ولده، يصير لعنة 🤡 - The conjuring 2 روح شريرة تستحوذ على بنت 😱 - The mist كائن مرعب يختفي وراء الضباب 😶 - Slasher قاتل متسلسل يهدد حياة الناس 😨  - OCULUS أخوات يحاولوا التغلب على صدمة وفاة والديهم'
predict2 = 'ابغى مسلسل اجوائه نفس  cable girls  و مراكش حُب وحرب وهذا ماجناه قلبي شي يحمس ويبسط يعني ☹️!!'
body = [
    {"text": predict},
    {"text": predict2}
]
r3 = requests.post(url + '/predict', json=body)
print(r3.content)
