import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

def search_title_name(name):
    title_search = re.search(' ([A-Za-z]+)\\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def encoder_colum(data_train,data_test,colum):
    encoder = LabelEncoder()
    data_train[colum] = encoder.fit_transform(data_train[colum])
    data_test[colum] = encoder.transform(data_test[colum])
    return data_train, data_test

models = {
    'Random Forest' : RandomForestClassifier(n_estimators = 200,max_depth = 3, criterion='entropy',random_state=0),
    'Decision Tree' : DecisionTreeClassifier(criterion='entropy',random_state=0),
    'Logistic' : LogisticRegression(max_iter=1000,random_state=0),
    'SVC' : SVC(kernel='linear', random_state=0),
    'K-Nearest' : KNeighborsClassifier(n_neighbors=5,metric='minkowski',p = 2),
    'Navie Bayes' : GaussianNB()
}
def choose_models(X_train,y_train,X_test,test_value):
    for name, model in models.items():
        model.fit(X_train,y_train)
        predictions_value = model.predict(X_test)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f'{name} Confusion Matrix:\n {confusion_matrix(test_value,predictions_value)}')
        print(f'{name} Classification Report:\n {classification_report(test_value,predictions_value)}')
        print(f'{name} Cross-validation scores: {scores}')
        print(f'{name} Mean cross-validation score: {scores.mean()}')
