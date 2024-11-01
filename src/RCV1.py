from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier

if __name__ == '__main__':
    print('Loading the RCV1 dataset. This may take a while...')
    X, y = fetch_rcv1(return_X_y = True)
    # Stratified split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                        random_state = 42)
    # Store the data in a binary format 
    save_npz('data/rcv1_train.npz', X_train)
    save_npz('data/rcv1_test.npz', X_test)
    save_npz('data/rcv1_train_labels.npz', y_train)
    save_npz('data/rcv1_test_labels.npz', y_test)

    # Train a SGD classifier (as baseline)
    clf = MultiOutputClassifier(SGDClassifier(random_state = 42))
    clf.fit(X_train, y_train.toarray())
    print('Baseline SGD model accuracy:', clf.score(X_test, y_test.toarray()))
