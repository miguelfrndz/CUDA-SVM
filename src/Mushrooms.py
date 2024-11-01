import os, sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if __name__ == "__main__":
    data = pd.read_table('data/mushrooms.csv', sep = ',', header = None)
    # Missing values are encoded as '?'
    data = data.apply(lambda x: x.replace('?', pd.NA))
    # Drop rows with missing values
    data = data.dropna()
    X, y = data.iloc[:, 1:], data.iloc[:, 0]
    # Encode y according to: edible=e, poisonous=p
    y = y.apply(lambda x: 1 if x == 'p' else 0)
    # Stratifed split into train and test and convert to numpy arrays
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    # Encode X using LabelEncoder
    le = LabelEncoder()
    X_train = X_train.apply(lambda x: le.fit_transform(x))
    X_test = X_test.apply(lambda x: le.fit_transform(x))
    # Save the processed data in the data folder
    train = pd.concat([X_train, y_train], axis = 1)
    test = pd.concat([X_test, y_test], axis = 1)
    # Save the processed data in the data folder as .data files
    train_file_path = 'data/mush_train.data'
    test_file_path = 'data/mush_test.data'

    with open(train_file_path, 'w') as train_file:
        train_file.write(f"{train.shape[0]} {train.shape[1] - 1}\n")
        train.to_csv(train_file, header=False, index=False, sep=' ')

    with open(test_file_path, 'w') as test_file:
        test_file.write(f"{test.shape[0]} {test.shape[1] - 1}\n")
        test.to_csv(test_file, header=False, index=False, sep=' ')
    
    print('Data processed and saved in data folder')

    # Baseline Scikit-Learn SVM model
    clf = SVC()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f'Baseline SVM model accuracy: {accuracy:.2f}')
    