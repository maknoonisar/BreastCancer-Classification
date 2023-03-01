"""
Breast Cancer Tumor Classification
Author: Guillermo Carsolio Gonzalez A01700041
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def get_accuracy_score(model, x_test, y_test):
    """
    Calculate Accuracy Score
    ---------------------------

    Parameters
    ----------

    model: sklearn.model,
        Sklearn model that was given
    x_test: pd.Dataframe(),
        x values of the testing dataset
    y_test: pd.Dataframe(),
        y values of the testing dataset

    returns
    -------
    accuracy_score:
        accuarcy of the model with the given testing dataframe
    """
    y_test_prediction = model.predict(x_test)
    return accuracy_score(y_test, y_test_prediction)


def logistic_regression(x_train, y_train, x_test, y_test):
    """
    Logistic Regression
    ---------------------------

    Parameters
    ----------

    x_train: pd.Dataframe(),
        x values of the training dataset
    y_train: pd.Dataframe(),
        y values of the training dataset
    x_test: pd.Dataframe(),
        x values of the testing dataset
    y_test: pd.Dataframe(),
        y values of the testing dataset

    returns
    -------
    dict:
        regression: LogisticRegression,
            logistic regression model trained with the training datasets
        accuracy: float,
            accuracy value tested with the testing datasets
    """
    regression = LogisticRegression(max_iter=7000)
    regression.fit(x_train, y_train)
    return {
        'model': regression,
        'accuracy': get_accuracy_score(regression, x_test, y_test)
    }


def decision_tree(x_train, y_train, x_test, y_test):
    """
    Decision Tree
    ---------------------------

    Parameters
    ----------

    x_train: pd.Dataframe(),
        x values of the training dataset
    y_train: pd.Dataframe(),
        y values of the training dataset
    x_test: pd.Dataframe(),
        x values of the testing dataset
    y_test: pd.Dataframe(),
        y values of the testing dataset

    returns
    -------
    dict:
        tree: DecisionTreeClassifier,
            decision tree model trained with the training datasets
        accuracy: float,
            accuracy value tested with the testing datasets
    """
    tree_m = DecisionTreeClassifier(max_depth=3)
    tree_m.fit(x_train, y_train)
    return {
        'model': tree_m,
        'accuracy': get_accuracy_score(tree_m, x_test, y_test)
    }


def random_forest(x_train, y_train, x_test, y_test):
    """
    Random Forest
    ---------------------------

    Parameters
    ----------

    x_train: pd.Dataframe(),
        x values of the training dataset
    y_train: pd.Dataframe(),
        y values of the training dataset
    x_test: pd.Dataframe(),
        x values of the testing dataset
    y_test: pd.Dataframe(),
        y values of the testing dataset

    returns
    -------
    dict:
        r_forest: RandomForestClassifier,
            random forest model trained with the training datasets
        accuracy: float,
            accuracy value tested with the testing datasets
    """
    r_forest = RandomForestClassifier(
        max_depth=40,
        min_samples_leaf=2,
        min_samples_split=17,
        n_estimators=1400
    )
    r_forest.fit(x_train, y_train)
    return {
        'model': r_forest,
        'accuracy': get_accuracy_score(r_forest, x_test, y_test)
    }


def df_cleaning(df, cols, char):
    """
    Dataframe Cleaning Function
    ---------------------------

    Parameters
    ----------

    df: pd.Dataframe(),
        This is the dataframe that will have unwanted data removed from

    cols: list,
        List of columns that want to be checked

    char: str,
        String with the unwanted value

    returns
    -------
    df: pd.Datafarme(),
        The original dataframe with the unwanted instances removed
    """
    for col in cols:
        df = df[df[col] != char]
    return df


def get_models(x_train, y_train, x_test, y_test):
    """
    Get all models
    ---------------------------
    This function does a call for each model and stores it in a dictonary.
    Also it presents to the user the accuarcy of each model and says the
    best model

    Parameters
    ----------

    x_train: pd.Dataframe(),
        x values of the training dataset
    y_train: pd.Dataframe(),
        y values of the training dataset
    x_test: pd.Dataframe(),
        x values of the testing dataset
    y_test: pd.Dataframe(),
        y values of the testing dataset

    returns
    -------
    models: dict,
        Dictonary of models
    """

    print('... Training models\n\n')
    models = {
        'Random Forest': random_forest(x_train, y_train, x_test, y_test),
        'Decision Tree': decision_tree(x_train, y_train, x_test, y_test),
        'Logistic Regression': logistic_regression(x_train, y_train, x_test, y_test)
    }
    highest = 0
    highest_model = ''
    for mod in models:
        print(mod + ' --- Accuracy: ' + str(models[mod]['accuracy']))
        if models[mod]['accuracy'] >= highest:
            highest = models[mod]['accuracy']
            highest_model = mod

    print('\nThe best model is --- ' + highest_model + '\n')
    return models


def get_value(attribute: str):
    """
    UI input attribute retrival
    ---------------------------

    Parameters
    ----------

    attribute: str,
        attribute that the user will input its designated value

    returns
    -------
    value: int,
        value that the user inputed into the command line within the wanted
        range
    """
    while True:
        value = int(input('What is the ' + attribute + '? [INT] 1-10:\n'))
        if value >= 1 and value <= 10:
            break
        else:
            print('OUT OF RANGE --- try again')
    return value


def predict_with_model(models, cols):
    """
    Predict User Input With Models
    ---------------------------
    In this function the program will interact with the user and predict
    given the models

    Parameters
    ----------

    models: dict,
        Dictonary of trained models

    cols: list,
        Attributes that the user will give the values too

    returns
    -------
    want_to_predict: bool,
        bit value that says if the user wants to keep predicting
    """
    user_values = []
    for col in cols:
        user_values.append(get_value(col))

    while True:
        want_multiple_models = input(
            'Do you want to be shown the results of all models? '
            'Y/ yes, N/ no: \n').lower()
        if want_multiple_models == 'n':
            while True:
                selection = int(input(
                    'What model do you want to use \n'
                    '1. Random Forest\n2. Decision Tree\n'
                    '3. Logistic Regression:\n'))
                if selection == 1 or selection == 2 or selection == 3:
                    if selection == 1:
                        print('\nUsing --- Random Forest')
                        print('Tumor is predicted to be: ' +
                              str(models[
                                  'Random Forest'
                              ]['model'].predict([user_values])))
                    elif selection == 2:
                        print('\nUsing --- Decision Tree')
                        print('Tumor is predicted to be: ' +
                              str(models[
                                  'Decision Tree'
                              ]['model'].predict([user_values])))
                    elif selection == 3:
                        print('\nUsing --- Logistic Regression')
                        print('Tumor is predicted to be: ' +
                              str(models[
                                  'Logistic Regression'
                              ]['model'].predict([user_values])))

                    break
                else:
                    print('INVALID INPUT --- try again')
            break

        elif want_multiple_models == 'y':
            for model in models:
                print('\n' + model + ' --- Tumor is predicted to be: ' +
                      str(models[model]['model'].predict([user_values])[0]))
            break
        else:
            print('INVALID INPUT --- try again')

    while True:
        want_to_predict = input(
            'Do you want to predict with again? Y/ yes, N/ no: \n').lower()
        if want_to_predict == 'n' or want_to_predict == 'y':
            break
        else:
            print('INVALID INPUT --- try again')
    return want_to_predict


def main():
    # Import datasets needed
    col = ['Clump Thickness', 'Uniformity of Cell Size',
           'Uniformity of Cell Shape', 'Marginal Adhesion',
           'Single Epithelial Cell Size', 'Bare Nuclei',
           'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    df = pd.read_csv('cancer_data.csv', names=col)

    # Getting rid of unwanted instances
    df = df_cleaning(df, col, '?')
    df['Class'] = df['Class'].replace([2], 'Benign')
    df['Class'] = df['Class'].replace([4], 'Malignant')

    # Splitting the Dataset into a training data set and a testing one
    df_x = df.loc[:, df.columns != 'Class']
    df_y = df['Class']
    x_train, x_test, y_train, y_test = train_test_split(
        df_x, df_y, test_size=0.25)

    # Retriving models
    models = get_models(x_train, y_train, x_test, y_test)

    # Starting user interface
    while True:
        if predict_with_model(models, col[:-1]) == 'n':
            break


main()
