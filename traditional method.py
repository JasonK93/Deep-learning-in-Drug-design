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

def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    clf = RandomForestClassifier(n_estimators=100, max_depth= 100).fit(X_train, y_train)
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
    randomforest()
    NB()


