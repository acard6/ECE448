'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    #raise RuntimeError("You need to write this part!")
    M, N = model.W.shape
    P = np.zeros((M, N, 4, M, N))
    for r in range(M):
        for c in range(N):
            if (model.T[r,c]):
                P[r,c,:,:,:] = 0
                continue
            for a in range(4):
                r2 = r
                c2 = c
                rl = r
                cl = c
                rr = r
                cr = c 
                if a == 0:
                    c2 = max(0, c-1)
                    rl = min(M-1, r+1)
                    rr = max(0, r-1)
                if a == 1:
                    r2 = max(0, r-1)
                    cl = max(0, c-1)
                    cr = min(N-1, c+1)
                if a == 2:
                    c2 = min(N-1, c+1)
                    rl = max(0, r-1)
                    rr = min(M-1, r+1)
                if a == 3:
                    r2 = min(M-1, r+1)
                    cl = min(N-1, c+1)
                    cr = max(0, c-1)
                
                if model.W[r2, c2]:
                    P[r,c,a,r,c] += model.D[r,c,0]
                else:
                    P[r,c,a,r2,c2] += model.D[r,c,0]
                if model.W[rl,cl]:
                    P[r,c,a,r,c] += model.D[r,c,1]    
                else:
                    P[r,c,a,rl,cl] += model.D[r,c,1]
                if model.W[rr,cr]:
                    P[r,c,a,r,c] += model.D[r,c,2]
                else:
                    P[r,c,a,rr,cr] += model.D[r,c,2]
    return P

def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    #raise RuntimeError("You need to write this part!")
    M,N = U_current.shape
    U_next = np.zeros((M,N))
    for r in range(M):
        for c in range(N):
            expected_utilities = []
            for a in range(4):
                #U_next[r,c] = model.R[r,c] + model.gamma * np.sum(P[r,c,a]*U_current)
                expected_utility = np.sum(P[r,c,a]*U_current)
                expected_utilities.append(expected_utility)
            U_next[r,c] = model.R[r,c] + model.gamma * np.max(expected_utilities)

    return U_next

def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    #raise RuntimeError("You need to write this part!")
    M,N = model.W.shape
    P = compute_transition_matrix(model)
    U_curr = np.zeros((M,N))
    U = np.zeros((M,N))
    for i in range(100):
        U = update_utility(model, P, U_curr)
        if np.max(np.abs(U - U_curr)) < epsilon:
            break
        U_curr = U
    return U

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
