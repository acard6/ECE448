'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''
    #images are ndarrays
    holding = {}
    label = []
    neighbors = []

    for i in range(len(train_images)):
        dist = np.sqrt(np.sum((image-train_images[i])**2))
        holding[dist] = i
    
    order = list(holding.keys())
    order.sort()    
    
    for i in range(k):
        idx = holding[order[i]]
        neighbors.append(train_images[idx])
        label.append(train_labels[idx])
    
    return np.array(neighbors), np.array(label)


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    hypotheses =[]
    scores = []

    for i in range(len(dev_images)):
        neigbors, label = k_nearest_neighbors(dev_images[i], train_images, train_labels, k)
        t = 0
        f = 0
        for j in label: 
            if bool(j):
                t += 1
            else:
                f += 1
        if (t>f):
            hypotheses.append(1)
            scores.append(t)
        else:
            hypotheses.append(0)
            scores.append(f)

    return np.array(hypotheses), np.array(scores)


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    confusions = np.zeros((2,2))
    
    for i in range(len(hypotheses)):
        predicted = hypotheses[i]
        actual =  references[i]
        confusions[int(actual)][int(predicted)] += 1

    TN = confusions[0][0]
    FP = confusions[0][1]
    FN = confusions[1][0]
    TP = confusions[1][1]
    precision = TP / (TP + FP)
    recall = TP / (TP +FN)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2/ ( (1/recall) + (1/precision))
    

    return confusions, accuracy, f1 

