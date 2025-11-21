
from typing import Optional

import jax
import numpy as np
import jax.numpy as jnp
import jax.random as random
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from flax import linen as nn
from flax.training import train_state
import optax

from scipy.stats import binom
import matplotlib.pyplot as plt

from functools import partial
import itertools
from typing import Tuple
import time


def sbc(prior_samples, # (B, theta_dim)
        post_samples, # (B, n, theta_dim)
        ):
    
    B, n, d = post_samples.shape

    x = jnp.arange(n+1)
    x2 = jnp.repeat(x, 2)  
    
    rank_ecdf = r_ecdf(prior_samples=prior_samples, post_samples=post_samples) # (B, d)

    unif_cdf = jnp.arange(n+2) / (n+1)
    u2 = jnp.repeat(unif_cdf, 2)[1:-1]

    uu = binom.ppf(0.995, B, jnp.arange(1,n+2)/(n+1)) / B
    uu = jnp.repeat(uu, 2, axis=0)[:-1]
    u_u = jnp.concat([jnp.zeros(1), uu])

    ul = binom.ppf(0.005, B, jnp.arange(1,n+2)/(n+1)) / B
    ul = jnp.repeat(ul, 2, axis=0)[:-1]
    u_l = jnp.concat([jnp.zeros(1), ul])

    ud_u = u_u - u2
    ud_l = u_l - u2


    for dim in range(d):
       
        y2 = jnp.repeat(rank_ecdf[:, dim], 2, axis=0)[:-1]
        y2 = jnp.concat([jnp.zeros(1), y2])

        plt.fill_between(x2, u_u, u_l, label="approx 90% CI", color="grey", alpha=0.25)

        
        plt.plot(x2, u2, label="unif_cdf", color = "darkgrey") 
        plt.plot(x2, y2, label="r_ecdf") 

        plt.legend()
        plt.show()

        diff2 = y2-u2

        plt.fill_between(x2, ud_l, ud_u, color="grey", alpha=0.25)
        plt.plot(x2, diff2, label="ecdf deviation")
        plt.legend()
        plt.show()



def r_ecdf(prior_samples, # (B, theta_dim)
        post_samples, # (B, n, theta_dim)
        ):
    
    B, n, d = post_samples.shape
   
    samples = jnp.concatenate([prior_samples[:, None], post_samples], axis=1) # (B, n+1, theta_dim)
    idcs = jnp.argsort(samples, axis=1) # indices that sort samples, 0 is prior sample
    ranks = jnp.argsort(idcs, axis=1)[:, 0, :] # (B, theta_dim) # position of index 0 in idcs is rank of samples[0]

    vmaped_bc = jax.vmap(fun=partial(jnp.bincount, length=n+1), in_axes=1, out_axes=1) # count ranks 0, ..., n
    counts = vmaped_bc(ranks)

    rank_ecdf = jnp.cumsum(counts, axis=0) / B # (B, theta_dim)

    return rank_ecdf 

# This code is taken directly from https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py
# Minor modifications have been made make it JAX compatible
def c2st(
    keys: map,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> jnp.ndarray:
    """Classifier-based 2-sample test returning accuracy

    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClassifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionality of the
    samples X and Y.

    Args:
        X: Sample 1
        Y: Sample 2
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    if z_score:
        X_mean = jnp.mean(X, axis=0)
        X_std = jnp.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * random.normal(next(keys), X.shape)
        Y += noise_scale * random.normal(next(keys), Y.shape)

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=int(random.randint(next(keys), (), 0, 1e5)),
    )

    data = jnp.concatenate((X, Y))
    target = jnp.concatenate(
        (
            jnp.zeros((X.shape[0],)),
            jnp.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=int(random.randint(next(keys), (), 0, 1e5)))
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = jnp.mean(scores, keepdims=True)
    return scores



class MLP_bin_clf(nn.Module):
    hidden_dims: list

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray # (B, d)
                 ) -> jnp.ndarray:

        h = x
        for layer in self.hidden_dims:
            h = nn.Dense(layer)(h)
            h = nn.relu(h)

        logits = nn.Dense(2)(h) # (B, 2)
        p = nn.softmax(logits) # (B, 2)

        return p 
    

def create_train_state(key,
                       model: MLP_bin_clf,
                       in_shape: Tuple[int, int],
                       lr: float,
                       ) -> train_state.TrainState:
    
    init_params = model.init(key, jnp.zeros(in_shape))

    tx = optax.adam(lr)

    return train_state.TrainState.create(
        apply_fn=model.apply, 
        params=init_params,
        tx=tx
        )

@partial(jax.jit, donate_argnames="state")
def train_step(state: train_state.TrainState,
               X: jnp.ndarray, # (B, d)
               Y: jnp.ndarray, # (B, )
               ) -> train_state.TrainState:
    
    def loss_fn(params):

        p = state.apply_fn(params, X) # (B, 2)

        mask = jnp.column_stack((Y, 1-Y)) # (B, 2)

        log_prob = jnp.mean(jnp.log(p) * mask) # ()

        return - log_prob
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)

    return state, loss

def train_MPL_bin_clf(keys: map,
                      model: MLP_bin_clf, 
                      X: jnp.ndarray, # (N, d)
                      Y: jnp.ndarray, # (N, )
                      lr: int,
                      batch_size: int,
                      n_epochs: int):
    
    N, d = X.shape
    state = create_train_state(next(keys), model, (batch_size, d), lr)
    
    n_batches = N // batch_size

    losses = []
    
    for epoch in range(n_epochs):

        idcs = jax.random.permutation(next(keys), N)

        X = X[idcs]
        Y = Y[idcs]

        for b in range(n_batches):
            x = X[b*batch_size :(b+1)*batch_size]
            y = Y[b*batch_size : (b+1)*batch_size]

            state, loss = train_step(state, x, y)
            losses.append(loss)

        # if epoch % 10 == 0:
        #     print(f"epoch: {epoch}  train loss: {loss:.4f}")

    return state, losses

def train_and_score(keys: map,
                    model: MLP_bin_clf, 
                    X_tr: jnp.ndarray, 
                    Y_tr: jnp.ndarray, 
                    X_te: jnp.ndarray, 
                    Y_te: jnp.ndarray, 
                    lr: int,
                    batch_size: int,
                    n_epochs: int):
    
    state, losses = train_MPL_bin_clf(keys,
                                        model,
                                        X_tr,
                                        Y_tr,
                                        lr,
                                        batch_size,
                                        n_epochs)
    
    p = model.apply(state.params, X_te)

    Y_pred = jnp.where(p[:, 0]>=0.5, jnp.ones_like(Y_te), jnp.zeros_like(Y_te))

    n_te = Y_te.shape[0] 

    acc_te = 1 - jnp.sum(jnp.abs(Y_pred - Y_te)) / n_te

    return acc_te
                

def c2st_jax(keys: map,
             X1: jnp.ndarray,
             X2: jnp.ndarray,
             lr: float = 1e-3,
             n_epochs: int = 300,
             folds: int = 5):
    
    
    d = X1.shape[-1]
    
    model = MLP_bin_clf(hidden_dims=[10*d, 10*d])

    X = jnp.concat([X1, X2], axis=0)
    Y = jnp.concat([jnp.ones((X1.shape[0],)), jnp.zeros((X2.shape[0],))], axis=0)

    N = X.shape[0]
    idcs = jax.random.permutation(next(keys), N)

    X = X[idcs]
    Y = Y[idcs]

    fold_size = N // folds

    batch_size = (folds -1) * fold_size

    test_acs = []

    all_idcs = jnp.arange(N)

    for f in range(folds):

        X_te = X[f*fold_size : (f+1)*fold_size]
        Y_te = Y[f*fold_size : (f+1)*fold_size]

        tr_idcs = jnp.concat([all_idcs[:f*fold_size ], all_idcs[(f+1)*fold_size:]])
        X_tr = X[tr_idcs]
        Y_tr = Y[tr_idcs]

        acc = train_and_score(keys, model, X_tr, Y_tr, X_te, Y_te, lr, batch_size, n_epochs)

        test_acs.append(acc)

    score = jnp.mean(jnp.array(test_acs))

    return score


if __name__ == "__main__":

    root_key = random.key(2)
    keys = map(partial(random.fold_in, root_key), itertools.count())

    N, d = int(1e3), 10

    # X = random.normal(next(keys), (N, d))
    # # Y = random.gamma(next(keys), a = 1, shape = (N, d))
    # Y = random.normal(next(keys), shape = (N, d))

    # scores = c2st(keys, X, Y)
    # print(scores)
    # B, n, d = 10000, 100, 2

    # # prior = -random.gamma(next(keys), 1, (B, d))
    # prior = random.normal(next(keys), (B, d))
    # post = random.normal(next(keys), (B, n, d))

    # sbc(prior_samples=prior, post_samples=post) # (B, d)

    X1 = 0.99*random.normal(next(keys), (N, d))
    X2 = random.normal(next(keys), (N, d))

    score = c2st_jax(keys, X1, X2)
    print(score)

    score = c2st(keys, X1, X2)
    print(score)

