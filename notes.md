# Notes

- vamapping d MDNs keeps their parameters independent, but the training is affected by the summed losses
    - ADAM consideres the global gradient norm
    - this makes the optimization problem d times higher dimensional
    - this can be worth it if things scale up (d or n_samples), because then vmap is efficient, e.g., only initialize model once
    - could also just loop of the d easier problems

- of course the task of learning the ammortized MDN is more difficult as the dimensionality of x increases