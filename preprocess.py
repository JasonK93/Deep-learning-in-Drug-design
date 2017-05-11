import pandas as pd
import re


def get_data(data):
    re_get_target = re.compile(r' "')
    first_column = data.iloc[:, 0]
    first_column_array = []

    for i in first_column:
        ele = re_get_target.split(i)
        first_column_array.append(ele)
    first_column = pd.DataFrame(first_column_array)
    target = first_column.iloc[:, 0]


    smile = first_column.iloc[:, 1]
    temp = []
    for i in smile:
        temp.append(re.sub(r'"', '', i))
    smile = pd.DataFrame(temp)

    fingerprint = first_column.iloc[:, 2]
    temp = []
    for i in fingerprint:
        temp.append(re.sub(r'"', '', i))
    fingerprint = pd.DataFrame(temp)

    temp = []
    occ_1 = first_column.iloc[:, 3]
    for i in occ_1:
        temp.append(int(re.sub(r'"', '', i)))
    occ_1 = pd.DataFrame(temp)

    occ_rest = data.iloc[:, 1:1023]

    last_column = data.iloc[:,1023]
    last_column_array = []

    for i in last_column:
        ele = re_get_target.split(i)
        last_column_array.append(ele)
    last_column = pd.DataFrame(last_column_array)

    occ_last = last_column.iloc[:, 0]
    temp = []
    for i in occ_last:
        temp.append(int(re.sub(r'"', '', i)))
    occ_last = pd.DataFrame(temp)

    value = last_column.iloc[:, 1]
    temp = []
    for i in value:
        temp.append(re.sub(r'"', '', i))
    value = pd.DataFrame(temp)

    method = last_column.iloc[:, 2]
    temp = []
    for i in method:
        temp.append(re.sub(r'"', '', i))
    method = pd.DataFrame(temp)

    dataset = []
    for i in xrange(0, len(value)):
        temp = []
        temp.append(target.iloc[i, ])
        temp.append(smile.iloc[i, ])
        temp.append(fingerprint.iloc[i, ])
        temp.append(list(occ_1.iloc[i, ]) + list(occ_rest.iloc[i, ]) + list(occ_last.iloc[i, ]))
        # temp.append(list(occ_rest.iloc[i, ]))
        # temp.append(list(occ_last.iloc[i, ]))
        temp.append(value.iloc[i, ])
        temp.append(method.iloc[i, ])
        dataset.append(temp)
        if i % 1000 == 0:
            print ('integration process :', i*100.0/len(value), '%')
    dataset = pd.DataFrame(dataset)
    return dataset


def get_X(data):
    X = []
    occp = data.iloc[:, 3]
    for i in xrange(0,len(data)):
        a = data.iloc[i, 2][0]
        c = re.findall('\d', a)
        d = []
        for k in xrange(0, len(c)):
            d.append(int(c[k]))
        for l in occp[i]:
            d.append(l)
        X.append(d)
    X = pd.DataFrame(X)
    return X


def get_value(data):
    a = data.iloc[:,4]
    value = []
    for i in xrange(0, len(data)):
        value.append(float(a[i][0]))
    return value


def get_target(data):
    a = data.iloc[:, 0]
    target = []
    for i in xrange(0, len(data)):
        target.append((a[i][0]))
    return target

def comb(X):
    test = []
    for i in xrange(0,len(X)):
        a = []
        temp = X.iloc[i, :]
        for k in xrange(0, 1024):
            a.append(temp[k])
            a.append(temp[k+1024])
        test.append(a)
        if i % 200 == 0:
            print ('combine preocess :', i*100/len(X), '%')
    comb = pd.DataFrame(test)
    return comb