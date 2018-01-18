import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from deeplearning.helpers import L_layer_model, L_model_forward

DATA_DIR = './data/'
PLOT = True


def main():
    # analyze_data()
    train, test, y_train = dummy_prepare_data()
    X_train = train.values.T
    X_test = test.values.T
    Y_train = y_train.values.T.reshape(1, y_train.shape[0])
    params, accuracy = dummy_train_model(X_train, Y_train)


def dummy_train_model(X_train, Y_train):

    # hyperparameters choices
    nb_of_hidden_layouts = 2
    nb_of_units_per_hidden_layouts = 8

    hidden_lay_dims = [nb_of_units_per_hidden_layouts] * nb_of_hidden_layouts
    layers_dims = (X_train.shape[0], *hidden_lay_dims, 1)

    # Reshape the training and test examples
    print('X_train shape', X_train.shape)
    print('Y_train shape', Y_train.shape)

    opti_params = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.01, num_iterations=30000, lambda_reg=0.4,
                                print_cost=True)

    y_train_hat, cache = L_model_forward(X_train, opti_params)
    train_predictions = y_train_hat > 0.5
    accuracy = np.sum(train_predictions == Y_train) / X_train.shape[0]
    print("train accuracy:", accuracy)

    return opti_params, accuracy


def dummy_fill_nan_values(df, display=True):
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


def dummy_prepare_data():
    train = pd.read_csv(DATA_DIR + "train.csv")
    test = pd.read_csv(DATA_DIR + "test.csv")
    # dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

    # Fill empty and NaNs values with NaN
    train = train.fillna(np.nan)
    test = test.fillna(np.nan)

    combine = pd.concat([train.drop('Survived', 1), test])
    combine['cabin_known'] = combine['Cabin'].isnull() == False
    combine['is_female'] = combine['Sex'] == 'female'

    # remove all non-quantitative data
    combine = combine.drop(['PassengerId', 'Cabin', 'Embarked', 'Name', 'Ticket', 'Sex'], 1)
    combine = dummy_fill_nan_values(combine, display=False)

    y_train = train['Survived']
    test = combine.iloc[len(train):]
    train = combine.iloc[:len(train)]

    return train, test, y_train


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


    #train_and_test = pd.concat([train.drop('Survived', 1), test])

if __name__ == '__main__':
    main()
