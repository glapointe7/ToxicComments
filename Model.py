import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_predictions(classifier, X_train_vect, X_test_vect, y_train, y_test, output_classes):
    roc_auc = []
    predictions = np.zeros(shape=(len(y_test), len(output_classes)))
    predictions_int = np.zeros(shape=(len(y_test), len(output_classes)))
    
    for i, output_class in enumerate(output_classes):
        classifier.fit(X_train_vect, y_train[output_class])
        
        predictions[:, i] = classifier.predict_proba(X_test_vect)[:, 1]
        predictions_int[:, i] = classifier.predict(X_test_vect)
        auc = roc_auc_score(y_test[output_class], predictions[:, i])
        roc_auc.append(auc)
        print("\nClass: ", output_class)
        print("ROC AUC: ", auc)

    print("\nMulti-class ROC AUC: ", np.mean(roc_auc))
    
    return roc_auc, predictions, predictions_int


def multi_class_rocauc(y_test, predictions):
    total_roc_auc = 0
    number_of_classes = len(y_test[0])
    for j in range(number_of_classes):
        total_roc_auc += roc_auc_score(y_test[:, j], predictions[:, j])

    return total_roc_auc / number_of_classes