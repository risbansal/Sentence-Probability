import re
import argparse

def read_doc():


    # readind data and counting words
    corpus = open("ngram_corpus.txt", 'r')  # opening corpus file
    corpus_data = corpus.read()  # readind data from corpus
    corpus_data1 = corpus_data.split()
    corpus_data2 = re.sub('[^A-Za-z]', ' ', corpus_data)  # Removing punctuations and special charachters
    corpus_data3 = re.sub('\s+', ' ', corpus_data2)
    corpus.close()


    return corpus_data1, corpus_data2, corpus_data3


def find_count(corpus_data1,corpus_data3, sentence1, sentence2):
    sentence_count = 0
    types = 0
    count = 0
    sentence1_count = []
    sentence2_count = []
    vocab_list = []

    corpus_list = corpus_data3.split()

    for word in corpus_data1:
        if word == ".":
            sentence_count += 1

    for word in corpus_list:
        if word not in vocab_list:
            vocab_list.append(word)
            types += 1


    # Finding occurances of words in sentence 1
    for i in range(len(sentence1)):
        for word in corpus_list:
            if (word == sentence1[i]):
                count += 1
        sentence1_count.append(count)

    # Finding occurances of words in sentence 2
    for i in range(len(sentence2)):
        for word in corpus_list:
            if (word == sentence2[i]):
                count += 1
        sentence2_count.append(count)

    return sentence_count, types, sentence1_count, sentence2_count, corpus_list


def initialization(sentence1, sentence2):
    #Initializing count matrix 1
    count_matrix1 = [[0] * len(sentence1) for i in range(len(sentence1))]
    for i in range(len(sentence1)):
        for j in range(len(sentence1)):
            count_matrix1[i][j] = 0

    #Initializing count matrix 2
    count_matrix2 = [[0] * len(sentence2) for i in range(len(sentence2))]
    for i in range(len(sentence2)):
        for j in range(len(sentence2)):
            count_matrix2[i][j] = 0


    #Initializing probability matrix 1
    prob_matrix1 = [[0] * len(sentence1) for i in range(len(sentence1))]
    for i in range(len(sentence1)):
        for j in range(len(sentence1)):
            prob_matrix1[i][j] = 0

    #Initializing probability matrix 2
    prob_matrix2 = [[0] * len(sentence2) for i in range(len(sentence2))]
    for i in range(len(sentence2)):
        for j in range(len(sentence2)):
            prob_matrix2[i][j] = 0

    return count_matrix1, count_matrix2, prob_matrix1, prob_matrix2

def bigram_prob(sentence1, sentence2, sentence_count, corpus_list, corpus_data1, count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, sentence1_count, sentence2_count):

    first_word1 = 0
    first_word2 = 0

    #Bigram count for Sentence 1
    for word in range(len(corpus_list)):
        for w in range(1, len(sentence1)):
            if sentence1[w] == corpus_list[word]:
                if sentence1[w-1] == corpus_list[word-1]:
                    count_matrix1[w-1][w] += 1

            if sentence1[w] == corpus_list[word]:
                if sentence1[w] == corpus_list[word-1]:
                    count_matrix1[w][w] += 1

            if sentence1[0] == corpus_list[word]:
                if sentence1[0] == corpus_list[word-1]:
                    count_matrix1[0][0] += 1

            if sentence1[0] == corpus_data1[word]:
                if "." == corpus_data1[word-1]:
                    first_word1 += 1

    #Bigram count for Sentence 2
    for word in range(len(corpus_list)):
        for w in range(1,len(sentence2)):
            if sentence2[w] == corpus_list[word]:
                if sentence2[w-1] == corpus_list[word-1]:
                    count_matrix2[w-1][w] += 1

            if sentence2[w] == corpus_list[word]:
                if sentence2[w] == corpus_list[word-1]:
                    count_matrix2[w][w] += 1

            if sentence2[0] == corpus_list[word]:
                if sentence2[0] == corpus_list[word-1]:
                    count_matrix2[0][0] += 1

            if sentence2[0] == corpus_data1[word]:
                if "." == corpus_data1[word-1]:
                    first_word2 += 1


    #Probability Matrix 1
    for row in range(len(sentence1)):
        for col in range(len(sentence1)):
            prob_matrix1[row][col] = (count_matrix1[row][col]/sentence1_count[col])

    #Probability matrix 2
    for row in range(len(sentence2)):
        for col in range(len(sentence2)):
            prob_matrix2[row][col] = (count_matrix2[row][col]/sentence2_count[col])

    #Probability of sentence1
    prob_s1 = first_word1/sentence_count
    for i in range(1,len(sentence1)):
        prob_s1 = prob_s1 * prob_matrix1[i-1][i]

    #Probability of sentence2
    prob_s2 = first_word2/sentence_count
    for i in range(1,len(sentence2)):
        prob_s2 = prob_s2 * prob_matrix2[i-1][i]


    return count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2


def trigram_prob(sentence1, sentence2, sentence_count, corpus_list, corpus_data1, count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, sentence1_count, sentence2_count):

    first_word1 = 0
    first_word2 = 0

    #Bigram count for Sentence 1
    for word in range(len(corpus_list)):
        for w in range(1, len(sentence1)):
            if sentence1[w] == corpus_list[word]:
                if sentence1[w-1] == corpus_list[word-1]:
                    if sentence1[w - 2] == corpus_list[word - 2]:
                        count_matrix1[w-1][w] += 1

            if sentence1[w] == corpus_list[word]:
                if sentence1[w] == corpus_list[word-1]:
                    if sentence1[w] == corpus_list[word - 2]:
                        count_matrix1[w][w] += 1

            if sentence1[0] == corpus_list[word]:
                if sentence1[0] == corpus_list[word-1]:
                    if sentence1[0] == corpus_list[word - 2]:
                        count_matrix1[0][0] += 1

            if sentence1[0] == corpus_data1[word]:
                if "." == corpus_data1[word-1]:
                    first_word1 += 1

    #Bigram count for Sentence 2
    for word in range(len(corpus_list)):
        for w in range(1,len(sentence2)):
            if sentence2[w] == corpus_list[word]:
                if sentence2[w-1] == corpus_list[word-1]:
                    if sentence2[w - 2] == corpus_list[word - 2]:
                        count_matrix2[w-1][w] += 1

            if sentence2[w] == corpus_list[word]:
                if sentence2[w] == corpus_list[word-1]:
                    if sentence2[w] == corpus_list[word - 2]:
                        count_matrix2[w][w] += 1

            if sentence2[0] == corpus_list[word]:
                if sentence2[0] == corpus_list[word-1]:
                    if sentence2[0] == corpus_list[word - 2]:
                        count_matrix2[0][0] += 1

            if sentence2[0] == corpus_data1[word]:
                if "." == corpus_data1[word-1]:
                    first_word2 += 1


    #Probability Matrix 1
    for row in range(len(sentence1)):
        for col in range(len(sentence1)):
            prob_matrix1[row][col] = (count_matrix1[row][col]/sentence1_count[col])

    #Probability matrix 2
    for row in range(len(sentence2)):
        for col in range(len(sentence2)):
            prob_matrix2[row][col] = (count_matrix2[row][col]/sentence2_count[col])

    #Probability of sentence1
    prob_s1 = first_word1/sentence_count
    for i in range(1,len(sentence1)):
        prob_s1 = prob_s1 * prob_matrix1[i-1][i]

    #Probability of sentence2
    prob_s2 = first_word2/sentence_count
    for i in range(1,len(sentence2)):
        prob_s2 = prob_s2 * prob_matrix2[i-1][i]


    return count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2


def display(count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2):
    print("Count Matrix for Sentence1")
    print(count_matrix1)
    print()

    print("Count Matrix for Sentence2")
    print(count_matrix2)
    print()

    print("Probability Matrix for Sentence1")
    print(prob_matrix1)
    print()

    print("Probability Matrix for Sentence2")
    print(prob_matrix2)
    print()

    print("Probability of Sentence1")
    print(prob_s1)
    print()

    print("Probability of Sentence2")
    print(prob_s2)
    print()

def smoothing(sentence1_count, sentence2_count, first_word1, first_word2, types, sentence1, sentence2, sentence_count, count_matrix1, count_matrix2, prob_matrix1, prob_matrix2):

    for i in range(len(sentence1_count)):
        sentence1_count[i] = sentence1_count[i] + types

    for row in range(len(sentence1)):
        for col in range(len(sentence1)):
            count_matrix1[row][col] += count_matrix1[row][col] + 1

    for i in range(len(sentence2_count)):
        sentence2_count[i] = sentence2_count[i] + types

    for row in range(len(sentence2)):
        for col in range(len(sentence2)):
            count_matrix2[row][col] = count_matrix2[row][col] + 1


    # Probability Matrix 1
    for row in range(len(sentence1)):
        for col in range(len(sentence1)):
            prob_matrix1[row][col] = (count_matrix1[row][col] / sentence1_count[col])

    # Probability matrix 2
    for row in range(len(sentence2)):
        for col in range(len(sentence2)):
            prob_matrix2[row][col] = (count_matrix2[row][col] / sentence2_count[col])

    # Probability of sentence1
    prob_s1 = (first_word1 + 1) / (sentence_count + types)
    for i in range(1, len(sentence1)):
        prob_s1 = prob_s1 * prob_matrix1[i - 1][i]

    # Probability of sentence2
    prob_s2 = (first_word2 + 1) / (sentence_count + types)
    for i in range(1, len(sentence2)):
        prob_s2 = prob_s2 * prob_matrix2[i - 1][i]

    return count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--ngram', type = int)
    parser.add_argument('-b', '--smoothing', type = int)
    arg = parser.parse_args()

    N = arg.ngram
    b = arg.smoothing


    sentence1 = "Milstein is a gifted violinist who creates all sorts of sounds and arrangements".split()
    sentence2 = "It was a strange and emotional thing to be at the opera on a Friday night".split()

    sentence1_count = []
    sentence2_count = []
    first_word1 = 0
    first_word2 = 0

    corpus_data1, corpus_data2, corpus_data3 = read_doc()
    sentence_count, types, sentence1_count, sentence2_count, corpus_list = find_count(corpus_data1, corpus_data3, sentence1, sentence2)
    count_matrix1, count_matrix2, prob_matrix1, prob_matrix2 = initialization(sentence1,sentence2)

    if N==2 and b==0:
        count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2 = bigram_prob(sentence1, sentence2, sentence_count, corpus_list, corpus_data1, count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, sentence1_count, sentence2_count)
        display(count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2)

    elif N==2 and b==1:
        count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2 = bigram_prob(sentence1, sentence2, sentence_count, corpus_list, corpus_data1, count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, sentence1_count, sentence2_count)
        count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2 = smoothing(sentence1_count, sentence2_count, first_word1, first_word2, types, sentence1, sentence2, sentence_count, count_matrix1, count_matrix2, prob_matrix1, prob_matrix2)
        display(count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2)

    elif N==3 and b == 0:
        count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2 = trigram_prob(sentence1, sentence2, sentence_count, corpus_list, corpus_data1, count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, sentence1_count, sentence2_count)
        display(count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2)

    elif N==3 and b == 1:
        count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2 = trigram_prob(sentence1, sentence2, sentence_count, corpus_list, corpus_data1, count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, sentence1_count, sentence2_count)
        count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2 = smoothing(sentence1_count, sentence2_count, first_word1, first_word2, types, sentence1, sentence2, sentence_count, count_matrix1, count_matrix2, prob_matrix1, prob_matrix2)
        display(count_matrix1, count_matrix2, prob_matrix1, prob_matrix2, prob_s1, prob_s2)

    else:
        print("Please input correct parameters. ie: -N{2,3} and -b{0,1}")


main()