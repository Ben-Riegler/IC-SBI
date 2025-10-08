# Notes

- vamapping d MDNs keeps their parameters independent, but the training is affected by the summed losses
    - ADAM consideres the global gradient norm
    - this makes the optimization problem d times higher dimensional
    - this can be worth it if things scale up (d or n_samples), because then vmap is efficient, e.g., only initialize model once
    - could also just loop of the d easier problems

- of course the task of learning the ammortized MDN is more difficult as the dimensionality of x increases

- only need F_Y(y|x) at selected values x and we can easily produce samples y|x for fixed x
    - approximate F_Y(y|x) can we constructed cheaply with ECF or softrank approx of it