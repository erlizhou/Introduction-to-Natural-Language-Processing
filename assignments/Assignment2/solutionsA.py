import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
	unigram_p = {}
	bigram_p = {}
	trigram_p = {}
	unigram_c, bigram_c, trigram_c = {}, {}, {}
	word_count = 0.0

	for sentence in training_corpus:
		tokens = sentence.strip().split()
		tokens.append(STOP_SYMBOL)
		for token in tokens:
			word_count += 1
			unigram_c[(token, )] = unigram_c.get((token, ), 0.0) + 1
		unigram_c[(START_SYMBOL, )] = len(training_corpus)
		tokens.insert(0, START_SYMBOL)
		for i in range(len(tokens) - 1):
			bigram_c[(tokens[i], tokens[i + 1])] = bigram_c.get((tokens[i], tokens[i + 1]), 0.0) + 1
		bigram_c[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
		tokens.insert(0, START_SYMBOL)
		for j in range(len(tokens) - 2):
			trigram_c[(tokens[j], tokens[j + 1], tokens[j + 2])] = trigram_c.get((tokens[j], tokens[j + 1], tokens[j + 2]), 0.0) + 1


	for word in unigram_c.keys():
		unigram_p[word] = math.log(unigram_c[word] / word_count, 2)
	for word in bigram_c.keys():
		if (word[0], ) in unigram_c:
			bigram_p[word] = math.log(bigram_c[word] / unigram_c[(word[0], )], 2)
		else:
			bigram_p[word] = MINUS_INFINITY_SENTENCE_LOG_PROB
	for word in trigram_c.keys():
		if (word[0], word[1]) in bigram_c:
			trigram_p[word] = math.log(trigram_c[word] / bigram_c[(word[0], word[1])], 2)
		else:
			trigram_p[word] = MINUS_INFINITY_SENTENCE_LOG_PROB

	return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    if n == 1:
    	for sentence in corpus:
    		tokens = sentence.strip().split()
    		tokens.append(STOP_SYMBOL)
    		prob = 0
    		for token in tokens:
    			prob += ngram_p[(token, )]
    		scores.append(prob)

    elif n == 2:
    	for sentence in corpus:
    		tokens = sentence.strip().split()
    		tokens.append(STOP_SYMBOL)
    		tokens.insert(0, START_SYMBOL)
    		prob = 0
    		for i in range(len(tokens) - 1):
    			prob += ngram_p[(tokens[i], tokens[i + 1])]
    		scores.append(prob)

    elif n == 3:
    	for sentence in corpus:
    		tokens = sentence.strip().split()
    		tokens.append(STOP_SYMBOL)
    		tokens.insert(0, START_SYMBOL)
    		tokens.insert(0, START_SYMBOL)
    		prob = 0
    		for i in range(len(tokens) - 2):
    			prob += ngram_p[(tokens[i], tokens[i + 1], tokens[i + 2])]
    		scores.append(prob)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    lamda = 1/3.0
    for sentence in corpus:
    	tokens = sentence.strip().split()
    	tokens.append(STOP_SYMBOL)
    	tokens.insert(0, START_SYMBOL)
    	tokens.insert(0, START_SYMBOL)
    	prob = 0

    	for i in range(3, len(tokens) + 1):
    		unigram = (tokens[i-1], )
    		bigram = (tokens[i-2], tokens[i-1])
    		trigram = (tokens[i-3], tokens[i-2], tokens[i-1])
    		if unigram not in unigrams or bigram not in bigrams or trigram not in trigrams:
    			prob = MINUS_INFINITY_SENTENCE_LOG_PROB
    			break
    		else:
    			prob += math.log(lamda * (2 ** unigrams[unigram] + 2 ** bigrams[bigram] + 2 ** trigrams[trigram]), 2)
    	scores.append(prob)

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
