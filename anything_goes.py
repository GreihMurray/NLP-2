import utility
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Work comes from https://nlpforhackers.io/training-pos-tagger/amp/

def anything():
    data = utility.read_file_to_sents()

    cutoff = int(.75 * len(data))
    training_sentences = data[:cutoff]
    test_sentences = data[cutoff:]

    X, y = transform_to_dataset(training_sentences)

    print(len(X))

    # all_clfs = dec_tree(X, y)
    # acc_score(all_clfs)

    model = seq_model(X, y)

def defeature(data):
    unfeatured = []

    for dict in data:
        unfeatured.append(dict['word'])

    return unfeatured


def string_to_num(data):
    le = LabelEncoder()

    label = le.fit_transform(data)

    return label


def seq_model(x, y):
    model = Sequential()
    model.add(Dense(256, input_shape=(1,), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    model.add(Dense(1, activation="softmax"))

    x = defeature(x)

    x = string_to_num(x)
    y = string_to_num(y)

    x = np.asarray(x)
    y = np.asarray(y)

    print("[INFO] training network...")
    sgd = SGD(0.01)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    H = model.fit(x, y, epochs=100, batch_size=32)

    return model

def dec_tree(X, y):
    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', DecisionTreeClassifier(criterion='entropy')),
    ], verbose=1)

    print("Training Started")

    # Custom work below
    all_clfs = []

    for i in tqdm(range(0, 115), desc="Training"):
        cur_clf = clf
        cur_clf.fit(X[(i * 31649): ((i + 1) * 31649)], y[(i * 31649): ((i + 1) * 31649)])  # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

        all_clfs.append(cur_clf)

    print('Training completed')

    X_test, y_test = transform_to_dataset(test_sentences)

    return all_clfs


def acc_score(all_clfs):
    all_scores = []

    for clf in all_clfs:
        all_scores.append(clf.score(X_test[:1500], y_test[:1500]))

    total_acc = 0

    for score in all_scores:
        total_acc += score

    print("Accuracy:", total_acc / len(all_scores))


# Comes from same link as above
def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y


if __name__ == '__main__':
    anything()