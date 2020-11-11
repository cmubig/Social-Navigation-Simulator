

import csv

ds = ['eth','hotel','univ','zara1','zara2']
data = []
for i in range(5):
    with open('result/'+ds[i]+'_error.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        data.append(list(spamreader)[1:])

with open('result/error.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(36):
        err = []
        for j in range(5):
            try:
                err.append(data[j][i][3])
                err.append(data[j][i][4])
                err.append(data[j][i][5])
            except:
                print(i,ds[j],j)
                err.append(None)
                err.append(None)
                err.append(None)
        spamwriter.writerow(err)
