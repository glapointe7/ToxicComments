import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


def calculate_predictions(classifier, X_train, X_test, y_train, y_test, output_classes):
    roc_auc = []
    predictions = np.zeros(shape=(len(y_test), len(output_classes)))
    predictions_int = np.zeros(shape=(len(y_test), len(output_classes)))
    
    for i, output_class in enumerate(output_classes):
        classifier.fit(X_train, y_train[output_class])
        
        predictions[:, i] = classifier.predict_proba(X_test)[:, 1]
        predictions_int[:, i] = classifier.predict(X_test)
        auc = roc_auc_score(y_test[output_class], predictions[:, i])
        roc_auc.append(auc)
        print("\nClass: ", output_class)
        print("ROC AUC: ", auc)

    print("\nMulti-class ROC AUC: ", np.mean(roc_auc))
    
    return roc_auc, predictions, predictions_int


def apply_best_model(dataset_name, train, test):
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X_train_vect = vectorizer.fit_transform(train.comment)
    X_test_vect = vectorizer.transform(test.comment)

    svm = LinearSVC(C=0.22)
    classifier = CalibratedClassifierCV(svm)

    classifier.fit(X_train_vect, train.is_toxic)

    predictions = classifier.predict_proba(X_test_vect)[:, 1]
    auc_roc = roc_auc_score(test.is_toxic, predictions)
    print(dataset_name + " AUC ROC: ", auc_roc)
    

def multi_class_rocauc(y_test, predictions):
    total_roc_auc = 0
    number_of_classes = len(y_test[0])
    for j in range(number_of_classes):
        total_roc_auc += roc_auc_score(y_test[:, j], predictions[:, j])

    return total_roc_auc / number_of_classes