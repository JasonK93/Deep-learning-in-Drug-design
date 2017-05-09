import pandas as pd
import preprocess


data = pd.read_csv("myFP_217_D2.csv", header=None)

D2 = preprocess.get_data(data)

X = preprocess.get_X(D2)
y = pd.DataFrame(preprocess.get_target(D2))


X_train, X_test = X[:int((0.8*len(X)))], X[int((0.8*len(X))):]
y_train, y_test = y[:int((0.8*len(X)))], y[int((0.8*len(X))):]

def svm():
    from sklearn import svm
    from sklearn.metrics import f1_score
    clf = svm.SVC().fit(X_train, y_train)
    # average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')
    accuracy = f1_score(y_test, clf.predict(X_test), average='macro')
    print 'SVM accuracy :' , accuracy
    print 'SVM trainset accuracy:', f1_score(y_train, clf.predict(X_train), average='macro')


def randomforest(n, depth):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    clf = RandomForestClassifier(n_estimators=n, max_depth= depth).fit(X_train, y_train)
    accuracy = f1_score(y_test, clf.predict(X_test), average='macro')
    print 'RF accuracy :' , accuracy
    print 'RF train accuracy:' ,f1_score(y_train, clf.predict(X_train), average='macro')


def NB():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import f1_score
    gnb = GaussianNB().fit(X_train, y_train)
    accuracy = f1_score(y_test, gnb.predict(X_test), average='macro')
    print 'NB accuracy :' , accuracy
    print 'NB train accuracy:', f1_score(y_train, gnb.predict(X_train), average='macro')



if __name__ == "__main__":
    svm()
    randomforest(100, 100)
    randomforest(100, 50)
    NB()

    X_less = pd.merge(X.iloc[:, 0:25], X.iloc[:, 1024:1049],how='outer', left_index=True, right_index=True)
    X_train, X_test = X_less[:int((0.8 * len(X)))], X_less[int((0.8 * len(X))):]
    randomforest(100,50)