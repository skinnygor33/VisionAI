from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

def apply_PCA(set_to_apply, n_comp):
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(set_to_apply)

def streamline_process(X_train, y_train):

    pipeline = Pipeline([
        ("classifier", LogisticRegression())
    ])

    parameters = {
        "classifier__penalty" : ['l2', 'elasticnet'],
        "classifier__tol" :  [float(x)/100000 for x in range(10)],
        "classifier__C" : [float(x)/2 for x in range(4)],
    }

    possible_models = GridSearchCV(
        pipeline,
        parameters,
        cv=None,
        scoring='f1'
    )

    possible_models = possible_models.fit(X_train, y_train)

    best_model = possible_models.best_estimator_

    return best_model, possible_models

def display_cm(best_model, X_test, y_test):

    y_predictions = best_model.named_steps["classifier"].predict(X_test)

    values_to_compare = confusion_matrix(y_test, y_predictions)

    display = ConfusionMatrixDisplay(confusion_matrix=values_to_compare, display_labels=['Legitimate', 'Fraud'])

    display.plot(cmap=plt.cm.Greens)
    plt.title('Confusion Matrix')
    plt.show()

def save_to_joblib_object(model, path):
    # Save the trained model to a file for future use
    model_filename = path + 'supervised_learning_model.pkl'
    joblib.dump(model, model_filename)