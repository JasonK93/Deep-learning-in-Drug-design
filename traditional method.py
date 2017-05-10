import pandas as pd
import preprocess
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

data = pd.read_csv("myFP_217_D2.csv", header=None)

D2 = preprocess.get_data(data)

X = preprocess.get_X(D2)
y = pd.DataFrame(preprocess.get_target(D2))

X, y = shuffle(X, y, random_state=0)

X_train, X_valid, X_test = X[:int((0.6*len(X)))], X[int((0.6*len(X))):int((0.8*len(X)))],  X[int((0.8*len(X))):]
y_train, y_valid, y_test = y[:int((0.6*len(X)))], y[int((0.6*len(X))):int((0.8*len(X)))],  y[int((0.8*len(X))):]




def svm():
    from sklearn import svm
    from sklearn.metrics import f1_score
    clf = svm.SVC().fit(X_train, y_train)
    # average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')
    accuracy = f1_score(y_test, clf.predict(X_test), average='macro')
    print 'SVM train set accuracy:', f1_score(y_train, clf.predict(X_train), average='macro')
    print 'SVM valid set accuracy:', f1_score(y_valid, clf.predict(X_valid), average='macro')
    print 'SVM test set accuracy:', f1_score(y_test, clf.predict(X_test), average='macro')


def randomforest(n, depth):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    clf = RandomForestClassifier(n_estimators=n, max_depth= depth).fit(X_train, y_train)
    accuracy = f1_score(y_test, clf.predict(X_test), average='macro')
    print 'RF train set accuracy:', f1_score(y_train, clf.predict(X_train), average='macro')
    print 'RF valid set accuracy:', f1_score(y_valid, clf.predict(X_valid), average='macro')
    print 'RF test set accuracy:', f1_score(y_test, clf.predict(X_test), average='macro')


def NB():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import f1_score
    gnb = GaussianNB().fit(X_train, y_train)
    accuracy = f1_score(y_test, gnb.predict(X_test), average='macro')
    print 'NB train set accuracy:', f1_score(y_train, gnb.predict(X_train), average='macro')
    print 'NB valid set accuracy:', f1_score(y_valid, gnb.predict(X_valid), average='macro')
    print 'NB test set accuracy:', f1_score(y_test, gnb.predict(X_test), average='macro')


def tree():
    from sklearn import tree
    clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
    # print 'tree train set accuracy:', f1_score(y_train, clf.predict(X_train), average='macro')
    # print 'tree valid set accuracy:', f1_score(y_valid, clf.predict(X_valid), average='macro')
    # print 'tree test set accuracy:', f1_score(y_test, clf.predict(X_test), average='macro')
    return f1_score(y_test, clf.predict(X_test), average='macro')

def less_X():
    X = pd.merge(X.iloc[:, 0:25], X.iloc[:, 1024:1049], how='outer', left_index=True, right_index=True)
    X, y = shuffle(X, y, random_state=0)

    X_train, X_valid, X_test = X[:int((0.6 * len(X)))], X[int((0.6 * len(X))):int((0.8 * len(X)))], X[int(
        (0.8 * len(X))):]
    y_train, y_valid, y_test = y[:int((0.6 * len(X)))], y[int((0.6 * len(X))):int((0.8 * len(X)))], y[int(
        (0.8 * len(X))):]

    randomforest(100, 50)



if __name__ == "__main__":
    svm()
    randomforest(100, 100)
    randomforest(100, 50)
    NB()
    less_X()

    plot_data = []
    for i in xrange(0, 100):
        plot_data.append(tree())
    plt.plot(plot_data)
