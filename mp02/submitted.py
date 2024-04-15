'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    #raise RuntimeError("You need to write this part!")
    d = {}
    for y in train:     #classes in the training set
        if y not in d:
            d[y] = Counter()
        for i in train[y]:     #ith text from class in training set
            for k in i: #kth word in the ith text from class in training set
                d[y][k] += 1
    return d


def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    #raise RuntimeError("You need to write this part!")
    nonstop = {}
    for y in frequency:
        nonstop[y] = Counter()
        for x in frequency[y]:
            if x not in stopwords:
                nonstop[y][x] = frequency[y][x]
    
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    #raise RuntimeError("You need to write this part!")
    likelihood = {}
   
    for y in nonstop:
        word_type = len(nonstop[y].keys())
        any_token = sum(nonstop[y].values()) # num. of tokens of any word in class y
        bottom = ((word_type + 1) * (smoothness)) + any_token
        likelihood[y] = {}
        for x in nonstop[y]:
            top = nonstop[y][x] + smoothness
            likelihood[y][x] = top / bottom
        likelihood[y]['OOV'] = smoothness / bottom
   
    return likelihood


def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    #raise RuntimeError("You need to write this part!")
    Ppos = np.log(prior)
    Pneg = np.log(1-prior)
    hypotheses = []

    for review in texts:
        totPos = 0
        totNeg = 0
        for x in review:
            for y in likelihood:
                tot = 0
                if x in stopwords:
                    continue
                elif x in likelihood[y]:
                    tot += np.log(likelihood[y][x])
                elif x not in likelihood[y]:
                    tot += np.log(likelihood[y]['OOV'])
                
                if y == 'pos':
                    totPos += tot
                else:
                    totNeg += tot
        if Ppos+totPos > Pneg+totNeg:
            hypotheses.append('pos')
        elif Ppos+totPos < Pneg+totNeg:
            hypotheses.append('neg')
        else:
            continue

    return hypotheses

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    #raise RuntimeError("You need to write this part!")
    accuracies = np.zeros(shape=(len(priors), len(smoothnesses))) #array[row][col]
    for P in range(len(priors)):
        for k in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[k])
            hypothesis = naive_bayes(texts, likelihood, priors[P])
            count = 0
            for (y,yhat) in zip(labels, hypothesis):
                if y == yhat:
                    count += 1
            correctness = (count / len(labels))
            
            accuracies[P][k] = correctness

    return accuracies