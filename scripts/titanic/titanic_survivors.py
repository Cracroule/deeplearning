import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from deeplearning.helpers import L_layer_model, L_model_forward, normalize_input

DATA_DIR = './data/'
EXPORT_DIR = './exports/'
PLOT = True


def main():
    # analyze_data()
    X_train, X_test, Y_train, cache_test = basic_prepare_data()

    assert X_train.shape[1] == Y_train.shape[1]  # sample size check
    np.random.seed(1)
    p = np.random.permutation(X_train.shape[1])  # we shuffle inputs
    X_train = X_train.T[p].T
    Y_train = Y_train.T[p].T

    # let s split our training data into 2 sets
    train_ratio = 0.8
    train_size = int(X_train.shape[1] * train_ratio)
    X_cross_val = X_train[:, train_size:]
    X_train = X_train[:, :train_size]
    Y_cross_val = Y_train[:, train_size:]
    Y_train = Y_train[:, :train_size]

    # train our model
    params, accuracy = basic_train_model(X_train, Y_train)
    print("train accuracy:", accuracy)

    y_cross_val_hat, _ = L_model_forward(X_cross_val, params)
    cross_val_predictions = y_cross_val_hat > 0.5
    accuracy = np.sum(cross_val_predictions == Y_cross_val) / X_cross_val.shape[1]
    print("cross val accuracy:", accuracy)

    y_test_hat, _ = L_model_forward(X_test, params)
    test_predictions = np.int8(y_test_hat > 0.5)

    res = pd.DataFrame({'PassengerId': cache_test.values.reshape(-1, ), 'Survived': test_predictions.T.reshape(-1, )})
    res.to_csv(EXPORT_DIR + "/submit.csv", index=False)


def basic_train_model(X_train, Y_train):

    # hyperparameters choices
    nb_of_hidden_layouts = 2
    nb_of_units_per_hidden_layouts = 10

    hidden_lay_dims = [nb_of_units_per_hidden_layouts] * nb_of_hidden_layouts
    layers_dims = (X_train.shape[0], *hidden_lay_dims, 1)

    # Reshape the training and test examples
    print('X_train shape', X_train.shape)
    print('Y_train shape', Y_train.shape)

    # the below gives a 77% accuracy on test set submited on kaggle (~ok accuracy, a bit less than avg submit)
    opti_params = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.5, num_iterations=8001, lambda_reg=0.2,
                                print_cost=True)

    y_train_hat, cache = L_model_forward(X_train, opti_params)
    train_predictions = y_train_hat > 0.5
    accuracy = np.sum(train_predictions == Y_train) / X_train.shape[1]

    return opti_params, accuracy


def basic_fill_nan_values(df, display=True):
    nul_nb_per_column = df.isnull().sum()
    if display:
        print("count of NaN data per column")
        print(nul_nb_per_column)
        print()
    input_names = list(df)
    for input_name in input_names:
        if nul_nb_per_column[input_name]:
            most_frequent_value = df[input_name].value_counts().idxmax()
            df[input_name] = df[input_name].fillna(most_frequent_value)
            if display:
                print(input_name, ' -> ', nul_nb_per_column[input_name], '(NaN values) filled with ->',
                      most_frequent_value)
    if display:
        print(df.info())
    return df


def basic_prepare_data():
    train = pd.read_csv(DATA_DIR + "train.csv")
    test = pd.read_csv(DATA_DIR + "test.csv")
    # dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

    # Fill empty and NaNs values with NaN
    train = train.fillna(np.nan)
    test = test.fillna(np.nan)

    cache_test = test["PassengerId"]  # need it later on to export results

    combine = pd.concat([train.drop('Survived', 1), test])
    combine['cabin_known'] = combine['Cabin'].isnull() == False
    combine['is_female'] = combine['Sex'] == 'female'

    # remove all non-quantitative data
    combine = combine.drop(['PassengerId', 'Cabin', 'Embarked', 'Name', 'Ticket', 'Sex'], 1)
    combine = basic_fill_nan_values(combine, display=False)

    y_train = train['Survived']
    Y_train = y_train.values.T.reshape(1, y_train.shape[0])
    standardized_combine, input_mean, input_var = normalize_input(combine.values.T)
    X_train = standardized_combine[:, :len(train)]
    X_test = standardized_combine[:, len(train):]

    return X_train, X_test, Y_train, cache_test


""" data analysis
some lines are commented but can be interesting to uncomment to see more graphs
This analysis is not directly used in data treatment, but ome decisions have been made considering it"""
def analyze_data():
    train = pd.read_csv(DATA_DIR + "train.csv")
    test = pd.read_csv(DATA_DIR + "test.csv")
    dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

    # Fill empty and NaNs values with NaN
    dataset = dataset.fillna(np.nan)
    train = train.fillna(np.nan)
    test = test.fillna(np.nan)

    # print(train.head(8))  # show the 8 first lines of the data
    # print(train.describe())  # provides statistics, column by column (quartiles, min, max, std)

    # print(train.isnull().sum())  # gives the number of null values per column
    # print(test.info())  # provides global info on the test_set

    # Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived
    # g = sns.heatmap(train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    # if PLOT: plt.show()

    surv = train[train['Survived'] == 1]
    nosurv = train[train['Survived'] == 0]
    surv_col = "blue"
    nosurv_col = "red"

    print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"
          % (len(surv), 1. * len(surv) / len(train) * 100.0,
             len(nosurv), 1. * len(nosurv) / len(train) * 100.0, len(train)))

    # plt.figure(figsize=[12, 10])
    # plt.subplot(331)
    # sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
    # sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col,
    #              axlabel='Age')
    # plt.subplot(332)
    # sns.barplot('Sex', 'Survived', data=train)
    # plt.subplot(333)
    # sns.barplot('Pclass', 'Survived', data=train)
    # plt.subplot(334)
    # sns.barplot('Embarked', 'Survived', data=train)
    # plt.subplot(335)
    # sns.barplot('SibSp', 'Survived', data=train)
    # plt.subplot(336)
    # sns.barplot('Parch', 'Survived', data=train)
    # # have to find a prettier way to display Fare
    # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
    #                     wspace=0.35)
    # if PLOT: plt.show()


    # AGE analysis
    g = sns.FacetGrid(train, col='Survived')
    g = g.map(sns.distplot, "Age")
    if PLOT: plt.show()
    # interesting -> young people survive more, old people survive less


    # FARE analysis
    # Fill Fare missing values with the median value (only one example of missing value here, so let s say it s fine)
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

    # g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f" % (dataset["Fare"].skew()))
    # g = g.legend(loc="best")
    # plt.show()
    # Fare distribution is very skewed. This can lead to overweigth very high values in the model, even if it is scaled
    # it is better to transform it with the log function to reduce this skew.

    # Apply log to Fare to reduce skewness distribution (rpil: any other fct such as power( ,1/4) could do well)
    # dataset["Fare"] = dataset["Fare"].map(lambda i: np.power(i, 1./4) if i > 0 else 0)
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f" % (dataset["Fare"].skew()))
    g = g.legend(loc="best")
    if PLOT: plt.show()


    # GENDER analysis
    g = sns.barplot(x="Sex", y="Survived", data=train)
    g = g.set_ylabel("Survival Probability")
    if PLOT: plt.show()
    print(train[["Sex", "Survived"]].groupby('Sex').mean())


    # PCLASS analysis
    plt.figure()
    plt.subplot(211)
    sns.barplot(x="Pclass", y="Survived", data=train, palette='muted')
    plt.subplot(212)
    sns.barplot(x="Pclass", y="Survived", data=train, hue='Sex', palette='muted')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    if PLOT: plt.show()

    # Embarked analysis
    dataset["Embarked"].isnull().sum()  # 2
    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    print(train.isnull().sum())
    train['Embarked'] = train["Embarked"].fillna("S")
    print(train.isnull().sum())

    g = sns.factorplot(x="Embarked", y="Survived", data=train,
                       size=6, kind="bar", palette="muted")
    g.despine(left=True)
    g = g.set_ylabels("survival probability")

    # Explore Pclass vs Embarked
    g = sns.factorplot("Pclass", col="Embarked", data=train,
                       size=6, kind="count", palette="muted")
    g.despine(left=True)
    g = g.set_ylabels("Count")
    if PLOT: plt.show()


if __name__ == '__main__':
    main()
