import utility
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

# Work comes from https://nlpforhackers.io/training-pos-tagger/amp/

def anything():
    data = utility.read_file_to_sents()

    cutoff = int(.75 * len(data))
    training_sentences = data[:cutoff]
    test_sentences = data[cutoff:]

    X, y = transform_to_dataset(training_sentences)

    print(len(X))

    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', DecisionTreeClassifier(criterion='entropy')),
    ], verbose=1)

    print("Training Started")

    clf.fit(X[:25000], y[:25000])  # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

    print('Training completed')

    X_test, y_test = transform_to_dataset(test_sentences)

    print("Accuracy:", clf.score(X_test[:1000], y_test[:1000]))


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