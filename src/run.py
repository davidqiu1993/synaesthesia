"""
run.py
The runner script for training and evaluation of the synaesthesia network model.
"""

__version__     = "1.0.0"
__author__      = "David Qiu"
__email__       = "david@davidqiu.com"
__website__     = "www.davidqiu.com"
__copyright__   = "Copyright (C) 2020, Dicong Qiu. All rights reserved."


import random
import numpy as np
import casadi as ca


# set random seed
RANDOM_SEED = 1000
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
#ca.random.seed(RANDOM_SEED) # cannot set random seed for casadi


# configure dimensions
N = [ 1, 1 ] # dimension of input for each modality
M = [ 1, 1 ] # dimension of output for each modality


def g(v):
    """
    Activation function.
    """

    s = 1 / (1 + ca.exp(-v))
    return s


def vectorize(x=None, W=None, K=None, s=None):
    """
    Vectorize variables.
    """

    x_vec = None
    if x is not None:
        x_vec = ca.vertcat(*x)

    W_vec = None
    if W is not None:
        W_vec = ca.vertcat(*[W_l.reshape((W_l.size1() * W_l.size2(), 1)) for W_l in W])

    K_vec = None
    if K is not None:
        K_vec = ca.vertcat(*[K_l.reshape((K_l.size1() * K_l.size2(), 1)) for K_l in K])

    s_vec = None
    if s is not None:
        s_vec = ca.vertcat(*s)

    return x_vec, W_vec, K_vec, s_vec


def build_network(N, M, g):
    """
    Build network.
    """

    L = len(N) # number of modalities

    # sanity check for dimension definitions
    assert(len(N) == L)
    assert(len(M) == L)

    # define symbolic variables
    x, W, K, s = [], [], [], []
    for l in range(L):
        x.append( ca.SX.sym('x' + str(l), N[l]) ) # input to each modality
        W.append( ca.SX.sym('W' + str(l), N[l], M[l]) ) # forward connections of each modality
        K.append( ca.SX.sym('K' + str(l), np.sum(M), M[l]) ) # cross-talk connections of echo modality
        s.append( ca.SX.sym('s' + str(l), M[l]) ) # output of each modality

    x_vec, W_vec, K_vec, s_vec = vectorize(x, W, K, s)

    # compute network output update
    s_update = []
    for l in range(L):
        s_l = []
        for i in range(M[l]):
            s_l_i = g( ca.mtimes(x[l].T, W[l][:,i]) + ca.mtimes(s_vec.T, K[l][:,i]) )
            s_l.append(s_l_i)
        s_l_vec = ca.vertcat(*s_l)
        s_update.append(s_l_vec)
    s_update_vec = ca.vertcat(*s_update)

    # compute sensitivity
    Chi = ca.jacobian(s_update_vec, x_vec)

    # compute objective
    loss = - ca.trace(ca.log(ca.mtimes(Chi.T, Chi)))

    # compute gradients
    dloss_dW = ca.gradient(loss, W_vec)
    dloss_dK = ca.gradient(loss, K_vec)

    # functionalize network output
    f_output = ca.Function(
        'output',
        [ x_vec, W_vec, K_vec, s_vec ],
        [ s_update_vec ]
    )

    # functionalize loss
    f_loss = ca.Function(
        'loss',
        [ x_vec, W_vec, K_vec, s_vec ],
        [ loss ]
    )

    # functionalize gradients
    f_dloss_dtheta = ca.Function(
        'dloss_dtheta',
        [ x_vec, W_vec, K_vec, s_vec ],
        [ dloss_dW, dloss_dK ]
    )

    return f_output, f_loss, f_dloss_dtheta


def initialize_network(N, M):
    """
    Initialize network.
    """

    L = len(N) # number of modalities

    # sanity check for dimension definitions
    assert(len(N) == L)
    assert(len(M) == L)

    # initialize network
    W0_val, K0_val = [], []
    for l in range(L):
        W0_val.append( ca.DM.rand(N[l], M[l]) ) # forward connections of each modality
        K0_val.append( ca.DM.rand(np.sum(M), M[l]) ) # cross-talk connections of echo modality

    return W0_val, K0_val


def generate_dataset(N, n=1):
    """
    Generate vectorized dataset.
    """

    N = np.sum(N)

    # generate randon configuration
    conf = []
    for d in range(N):
        conf.append((
            np.random.rand() * 20.0 - 10.0, # mean in [ -10.0, 10.0 ]
            1.0 + np.random.rand() * 2.0    # var in [ 1.0, 3.0 ]
        ))

    # generate dataset
    X = []
    for i in range(n):
        x = []
        for d in range(N):
            x.append( ca.DM(np.random.normal(conf[d][0], conf[d][1])) )
        X.append(ca.DM(x))

    return X, conf


def evaluate_network(f_loss, W_val_vec, K_val_vec, X_val_vec, S_val_vec):
    """
    Evaluate network.
    """

    losses = []
    for i, (x_val_vec, s_val_vec) in enumerate(zip(X_val_vec, S_val_vec)):
        loss = f_loss(x_val_vec, W_val_vec, K_val_vec, s_val_vec)
        losses.append(loss)

    average_loss = np.mean(losses)

    return average_loss


def train_network(M, f_output, f_loss, f_dloss_dtheta, W0_val_vec, K0_val_vec, s0_val_vec, X_val_vec, lr=0.001, batch_size=64, n_iter=1):
    """
    Train network.
    """

    hist = []

    # generate initial network output
    S0_val_vec = []
    for x_val_vec in X_val_vec:
        S0_val_vec.append(f_output(x_val_vec, W0_val_vec, K0_val_vec, s0_val_vec))

    # evaluate the initial network
    hist.append({
        'W_vec': ca.DM(W0_val_vec),
        'K_vec': ca.DM(K0_val_vec),
        'S_vec': [ ca.DM(s_val_vec) for s_val_vec in S0_val_vec ],
        'loss': evaluate_network(f_loss, W0_val_vec, K0_val_vec, X_val_vec, S0_val_vec),
    })
    print('[ init ] loss: %.6f' % (hist[-1]['loss']))

    # train and evaluate network
    for iter in range(n_iter):
        # make batches
        batches = []
        samples = []
        for sample in zip(X_val_vec, hist[-1]['S_vec']):
            samples.append(sample)
        random.shuffle(samples)
        for i_batch in range(len(samples) // batch_size):
            batches.append(samples[i_batch * batch_size : (i_batch + 1) * batch_size])
        if len(samples) % batch_size > 0:
            batches.append(samples[- len(samples) % batch_size :])

        # train with batches
        W_val_vec = hist[-1]['W_vec']
        K_val_vec = hist[-1]['K_vec']
        for batch in batches:
            # compute individual gradients for each batch sample
            dloss_dW_vals, dloss_dK_vals = [], []
            for x_val_vec, s_val_vec in batch:
                dloss_dW_val, dloss_dK_val = f_dloss_dtheta(
                    x_val_vec, hist[-1]['W_vec'], hist[-1]['K_vec'], s_val_vec
                )
                dloss_dW_vals.append(dloss_dW_val)
                dloss_dK_vals.append(dloss_dK_val)

            # compute weighted average gradients for the batch
            averge_dloss_dW_val = ca.sum2(ca.horzcat(*dloss_dW_vals)) / batch_size
            averge_dloss_dK_val = ca.sum2(ca.horzcat(*dloss_dK_vals)) / batch_size

            # update network
            W_val_vec = W_val_vec - lr * averge_dloss_dW_val
            K_val_vec = K_val_vec - lr * averge_dloss_dK_val

        # generate network output
        S_val_vec = []
        for x_val_vec, s_val_vec in zip(X_val_vec, hist[-1]['S_vec']):
            S_val_vec.append(f_output(x_val_vec, W_val_vec, K_val_vec, s_val_vec))

        # evaluate network
        hist.append({
            'W_vec': ca.DM(W_val_vec),
            'K_vec': ca.DM(K_val_vec),
            'S_vec': [ ca.DM(s_val_vec) for s_val_vec in S_val_vec ],
            'loss': evaluate_network(f_loss, W_val_vec, K_val_vec, X_val_vec, S_val_vec),
        })
        print('[ %d ] loss: %.6f' % (iter + 1, hist[-1]['loss']))

    return hist


def main():
    """
    Run training and evaluation.
    """

    # build network
    f_output, f_loss, f_dloss_dtheta = build_network(N, M, g)

    # initialize variables
    W0_val, K0_val = initialize_network(N, M)
    _, W0_val_vec, K0_val_vec, _ = vectorize(None, W0_val, K0_val, None)
    s0_val_vec = ca.DM.zeros(np.sum(M))

    # generate dataset
    X_val_vec, X_vec_conf = generate_dataset(N, n=1000)

    # train network
    hist = train_network(
        M, f_output, f_loss, f_dloss_dtheta, W0_val_vec, K0_val_vec, s0_val_vec, X_val_vec,
        lr=0.00035, batch_size=64, n_iter=1000
    )

    # visualize the learning curve
    import matplotlib.pyplot as plt

    losses = []
    filtered_losses = []
    filter_window_size = 5
    filter_window = [ hist[0]['loss'] for i in range(filter_window_size) ]

    for record in hist:
        losses.append(record['loss'])
        filter_window.append(record['loss'])
        filter_window = filter_window[-filter_window_size:]
        filtered_losses.append(np.mean(filter_window))

    plt.plot([iter for iter in range(len(losses))], losses, lw=0.3)
    plt.plot([iter for iter in range(len(filtered_losses))], filtered_losses, lw=1.0)
    plt.title('learning curve')
    plt.legend(['raw', 'filtered'])
    plt.xlabel('iter')
    plt.ylabel('loss')

    plt.show()


if __name__ == '__main__':
    main()
