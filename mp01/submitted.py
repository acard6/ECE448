'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    #raise RuntimeError('You need to write this part!')
    rows, cols = (len(texts), len(texts))
    pmf = [[0 for i in range(cols)] for j in range(rows)]
    
    total = 0
    maxW1 = 0
    maxW2 = 0
    for lists in texts:
      c0 = 0
      c1 = 0
      for word in lists:
        if word == word0:
          c0 += 1
        elif word == word1:
          c1 += 1
      if c0 >= maxW1:
        maxW1 = c0
      if c1 >= maxW2:
        maxW2 = c1          
      pmf[c0][c1] += 1
      total += 1

    temp = [[(pmf[j][i]/total) for i in range(maxW2+1)] for j in range(maxW1+1)]    
    del(pmf)

    Pjoint = np.array(temp)

    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    #raise RuntimeError('You need to write this part!')
    if index == 0: #sum up the columns
    #  temp = [0]*len(Pjoint[0])
    #  for i in range(len(Pjoint[0])): #for every column
    #      temp[i] += sum(Pjoint[i])
      temp = np.sum(Pjoint, axis=0)


    else:
      temp = [0]*len(Pjoint[0])
      for i in range(len(Pjoint[0])): #for every row in the list
        for j in range(len(Pjoint)): #for every column in a row
          temp[i] += Pjoint[j][i]
    
    Pmarginal = np.array(temp)
    return Pmarginal
    
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
   #raise RuntimeError('You need to write this part!')
    #rows, cols = (len(Pjoint[0]), len(Pjoint))
    #temp = [[0 for i in range(cols)] for j in range(rows)]
    Pcond = Pjoint
    k = 0
    for i in Pjoint:
      for j in range(len(i)):
          Pcond[k][j] =i[j] / Pmarginal[k]
      k += 1
    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''
    #raise RuntimeError('You need to write this part!')
    mu = 0
    for i in range(len(P)):
      mu += P[i] * i
    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    #raise RuntimeError('You need to write this part!')
    mu = mean_from_distribution(P)
    var = 0
    for i in range(len(P)):
      var += (i - mu) ** 2 * P[i]
    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    #raise RuntimeError('You need to write this part!')
    x = marginal_distribution_of_word_counts(P,1)
    mu_x = mean_from_distribution(x)
    y = marginal_distribution_of_word_counts(P,0)
    mu_y = mean_from_distribution(y)

    M = len(P)
    N = len(P[0])
    covar = sum([P[X][Y]*(Y-mu_y)*(X-mu_x) for X in range(M) for Y in range(N)])

    #for Y in range(len(P[0])):
    #  for X in range(Y):
    #    covar += (Y - mu_y) * (X - mu_x) * P[Y][X] 
    
    return covar

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    #raise RuntimeError('You need to write this part!')
  
    M = len(P)
    N = len(P[0])
    expected = sum([f(X,Y) * P[X][Y] for X in range(M) for Y in range(N)])
    #for Y in range(len(P[0])):
    #  for X in range(Y):
    #    expected += f(X,Y) * P[Y][X]
    
    return expected
    
