'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np
import utils

# define your epsilon for laplace smoothing here
k = 1e-6
def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    d = defaultdict(list)
    #tags = []
    for sentence in train:
        #sent_list = []
        for word, tag in sentence:
                #if word == 'START' or word == 'END':
                #        continue

                # data structure is as follows
                # defaultdict(<class 'list'>,{word:Counter({tag1:count, ....}), ...})
                if len(d[word]) == 0:
                     d[word] = Counter([tag]) 
                else:
                     d[word].update([tag])
    
        #tags.append(sent_list)
    comm_tag = {}
    tagNcount = {}
    for word in d:
        comm_tag[word] = d[word].most_common(1)[0][0]
        tagNcount[word] = d[word].most_common(1)[0]

    m = Counter()
    # the occurance of each tag in the total training set
    for i in tagNcount:
          m[tagNcount[i][0]] += tagNcount[i][1]
    #grabs the top tag in the entire trainign set
    most_comm_tag = m.most_common(1)[0][0] 

    #test_strip = utils.strip_tags(test)
    tagged_data = []
    for sentence in test:
        sent_list = []
        for word in sentence:
                if word in d:
                        sent_list.append((word, comm_tag[word]))
                else:
                        sent_list.append((word, most_comm_tag))
        tagged_data.append(sent_list)
    return tagged_data

def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # transitional prob: a_{i,j} = P(Y_{t}=j | Y_{t-1}=i) 
    # = Y[i][j] / sum(Y[i].values()), where Y is the defaultdict(list)
    # obseration prob: b_{j}(x_{t}) = P(X_{t}=x_{t} | Y_{t}=j) = P(w|t). the prob of a word given a tag
    # P(w|t) = twp[t][w] / sum(twp[t].values())

    # construct the hidden markov model for the viterbi
    init = Counter()            # counter for initial prob, counts the times a word starts the sentance
    d    = defaultdict(list)    # counter for transition prob, gives count relation between 2 proceeding tags
    twp  = defaultdict(list)    # counter for emission prob, counts the times a word shows up when a certain tag is used
    for sentence in train:
        for i in range(len(sentence)-1):
                word = sentence[i][0]
                tag = sentence[i][1]
                # prev_word = sentence[i+1][0]
                next_tag = sentence[i+1][1]

                # the number of times a certain tag shows up at the start of a sentence
                if i == 0:
                      init.update([tag]) 

                # counts how many times a certain tag proceeds the current tag 
                # basically counts occurance of tag tag
                if len(d[tag]) == 0:
                     d[tag] = Counter([next_tag]) 
                else:
                     d[tag].update([next_tag])

                if len(twp[tag]) == 0:
                      twp[tag] = Counter([word])
                else:
                      twp[tag].update([word])
        # edge case for END if it for some reason casues me issues. will comments out if it makes no difference
        if len(twp['END']) == 0:
              twp['END'] = Counter(['END'])
        else:
              twp['END'].update(['END'])
    # training should be completed
    lap_init = {}
    lap_tran = {}
    lap_emis = {}

    # the log of the smoothed initial probability
    total_laplace = (k*(len(init)+1) + sum(init.values()))
    for word in init:
        lap_init[word] = log((init[word]+k) / total_laplace)
    # placeholder value for any word not in our trainig vocab
    #lap_init['OOV'] = log(k/ total_laplace)
    
    # the log of the smoothed transitional probability
    for tag in d:
        lap_tran[tag] = {}  
        laplaced = (k*(len(d[tag])+1) + sum(d[tag].values()))
        for i in d[tag]:
                prob = log((d[tag][i]+k) / laplaced)
                lap_tran[tag][i] = prob
        lap_tran[tag]['OOV'] = log(k / laplaced)

    # the log of the smoothed emission probability
    for tag in twp:
        lap_emis[tag] = {}
        laplaced = (k*(len(twp[tag])+1) + sum(twp[tag].values()))
        for word in tag:
                prob = log((twp[tag][word]+k) / laplaced)
                lap_emis[tag][word] = prob
        lap_emis[tag]['OOV'] = log(k / laplaced)

    # construct trellis 
    tagged_data = []
    for sentence in test:
          tagged_sent = helper(sentence, lap_init, lap_tran, lap_emis)
          tagged_data.append(tagged_sent)
    
    return tagged_data


def helper(sentence, lap_init, lap_tran, lap_emis):
        '''
        helper funtction for the viterbi function.
        Given a sentance, initial state, transition, and observational prob
        calculated the trellis and returns the best path
        input:  sentence (list of words to be tagged). E.g., [word1, word2, ...]
                lap_init (dict of words and their log-probability of being seen at the begging of a sentance)
                lap_tran (dict of dict where each key is a tag with value of all proceeding tags seen in training and their log-probabilities). E.g., {tag1:{proTag1:-1.2, proTag2:-0.36}, ...}
                lap_emis (dict of dict where each key is a tag and the value is the log-probability of a words as seen in training). E.g., {tag1:{word1:-1.2, word2:-0.36}, ...}
        '''    
        T = len(sentence)
        trellis = [{}]
        for tag in lap_init:
              trellis[0][tag] = (lap_init[tag] + lap_emis[tag].get(sentence[0], lap_emis[tag]['OOV']), None)
        
        for t in range(1,T):
              trellis.append({})
              for tag in lap_tran:
                        if tag != 'START':
                                prob, prev_tag = max((trellis[t-1][prev_tag][0] + lap_tran[prev_tag][tag] + lap_emis[tag].get(sentence[t], lap_emis[tag]['OOV']), prev_tag) 
                                        for prev_tag in lap_tran if prev_tag in trellis[t-1])
                                trellis[t][tag] = (prob, prev_tag)

        best_prob, best_tag = max((trellis[T-1][tag][0], tag) for tag in trellis[T-1])
        best_path = [(sentence[T-1], best_tag)]

        for t in range(T-2, -1, -1):
                best_path.append((sentence[t], trellis[t+1][best_tag][1]))
                best_tag = trellis[t+1][best_tag][1]
        best_path.reverse()

        return best_path

def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



