from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import nltk
import numpy as np

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}
    word = None

    for lexelt in data.keys():
        vector = set()
        for sentences in data[lexelt]:
            token = []
            for sentence in sentences[1:-1]:
                newtoken = nltk.word_tokenize(sentence)
                token += newtoken
                if len(newtoken) == 1:
                    word = newtoken[0]
            index = token.index(word)

            if index < window_size:
            	for element in token[:index]:
            		vector.add(element)
            else:
            	for element in token[index - window_size:index]:
            		vector.add(element)

            if index > len(token) - 1 - window_size:
            	for element in token[index + 1:]:
            		vector.add(element)
            else:
            	for element in token[index + 1:index + 1 + window_size]:
            		vector.add(element)

        s[lexelt] = list(vector)

    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    for instance in data:
    	labels[instance[0]] = instance[-1]
    	vector = [0 for x in range(len(s))]
    	token = []
    	for context in instance[1:-1]:
    		token += nltk.word_tokenize(context)
    	for word in token:
    		if word in s:
    			index = s.index(word)
    			vector[index] += 1

    	vectors[instance[0]] = vector

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

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

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    count_vector = []
    sense_vector = []
    test_vector = []

    for key in sorted(y_train.keys()):
    	count_vector.append(X_train[key])
    	sense_vector.append(y_train[key])

    count_array = np.array(count_vector)
    sense_array = np.array(sense_vector)

    svm_clf.fit(count_array, sense_array)
    knn_clf.fit(count_array, sense_array)

    for key in sorted(X_test.keys()):
    	test_vector.append(X_test[key])

    test_array = np.array(test_vector)
    svm_predict = svm_clf.predict(test_array)
    knn_predict = knn_clf.predict(test_array)

    counter = 0
    for key in sorted(X_test.keys()):
    	svm_results.append((key, svm_predict[counter]))
    	knn_results.append((key, knn_predict[counter]))
        counter += 1

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''
    for key in sorted(results.keys()):
    	tuples = results[key]
    	for tup in tuples:
    		with open(output_file, 'a') as the_file:
    			the_file.write(replace_accented(key) + ' ' + replace_accented(tup[0]) + ' ' + replace_accented(tup[1].decode('unicode-escape')) + '\n')

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



