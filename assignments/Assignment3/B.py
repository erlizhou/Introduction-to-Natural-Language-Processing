import A
from sklearn.feature_extraction import DictVectorizer
import nltk
from sklearn import svm
import numpy as np


# You might change the window size
window_size = 3


# B.1.a,b,c,d
def extract_features(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    for instance in data:
        feature = dict()
        labels[instance[0]] = instance[-1]
        token = []
        for sentence in instance[1:-1]:
            newtoken = nltk.word_tokenize(sentence)
            token += newtoken
            if len(newtoken) == 1:
                    word = newtoken[0]
        index = token.index(word)

        if index < window_size:
            for element in token[:index]:
                feature[element] = feature.get(element, 0) + 1
        else:
            for element in token[index - window_size:index]:
                feature[element] = feature.get(element, 0) + 1

        if index > len(token) - 1 - window_size:
            for element in token[index + 1:]:
                feature[element] = feature.get(element, 0) + 1
        else:
            for element in token[index + 1:index + 1 + window_size]:
                feature[element] = feature.get(element, 0) + 1

        features[instance[0]] = feature

    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''



    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []


    svm_clf = svm.LinearSVC()
    

    count_vector = []
    sense_vector = []
    test_vector = []

    for key in sorted(y_train.keys()):
        count_vector.append(X_train[key])
        sense_vector.append(y_train[key])

    count_array = np.array(count_vector)
    sense_array = np.array(sense_vector)

    svm_clf.fit(count_array, sense_array)
    

    for key in sorted(X_test.keys()):
        test_vector.append(X_test[key])

    test_array = np.array(test_vector)
    svm_predict = svm_clf.predict(test_array)
    

    counter = 0
    for key in sorted(X_test.keys()):
        results.append((key, svm_predict[counter]))
        counter += 1

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)