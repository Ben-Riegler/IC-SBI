
from typing import Optional

import jax
import numpy as np
import jax.numpy as jnp
import jax.random as random
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from functools import partial
import itertools


def sbc(prior_samples, # (B, theta_dim)
        post_samples, # (B, n, theta_dim)
        ):
    
    B, n, d = post_samples.shape
    
    rank_ecdf = r_ecdf(prior_samples=prior_samples, post_samples=post_samples) # (B, d)

    for dim in range(d):
        x = jnp.arange(n+1)
        x2 = jnp.repeat(x, 2)[1:]          
        y2 = jnp.repeat(rank_ecdf[:, dim], 2, axis=0)[:-1]
        unif_cdf = jnp.arange(1, n+2) / (n+1)
        u2 = jnp.repeat(unif_cdf, 2)[:-1]

        plt.plot(x2, y2, label="r_ecdf") 
        plt.plot(x2, u2, label="unif_cdf") 
        plt.legend()
        plt.show()

        diff2 = y2-u2

        plt.plot(x2, diff2, label="ecdf deviation")
        plt.legend()
        plt.show()

    
    



def r_ecdf(prior_samples, # (B, theta_dim)
        post_samples, # (B, n, theta_dim)
        ):
    
    B, n, d = post_samples.shape
   
    samples = jnp.concatenate([prior_samples[:, None], post_samples], axis=1) # (B, n+1, theta_dim)
    idcs = jnp.argsort(samples, axis=1)
    ranks = jnp.argsort(idcs, axis=1)[:, 0, :] # (B, theta_dim)

    vmaped_bc = jax.vmap(fun=partial(jnp.bincount, length=n+1), in_axes=1, out_axes=1)
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


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    root_key = random.key(2)
    keys = map(partial(random.fold_in, root_key), itertools.count())

    # N, d = int(1e3), 10

    # X = random.normal(next(keys), (N, d))
    # # Y = random.gamma(next(keys), a = 1, shape = (N, d))
    # Y = random.normal(next(keys), shape = (N, d))

    # scores = c2st(keys, X, Y)
    # print(scores)
    B, n, d = 10000, 100, 2

    # prior = -random.gamma(next(keys), 1, (B, d))
    prior = 2 * random.normal(next(keys), (B, d))
    post = random.normal(next(keys), (B, n, d))


    sbc(prior_samples=prior, post_samples=post) # (B, d)

