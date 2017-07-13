
# Things learned about machine learning during this exercise:
- the initial weight values have a lot more influence than I first thought they would
- I think that as a rule of thumb: initialise hidden weights roughly of the order of magnitude that you want to work with (more flexible than the output weights), and the output layer should have weights roughly of the order of magnitude that you want the outputs to have
- ReLu gives the results a straight line feel whereas tanh gives the results a more curved feel.
    - It makes sense but I didn't expect it because I thought that since they both offer 'universal approximation', that their results would be identical
- ReLu doesn't work as well with MC-dropout as tanh (I think)
- functions are harder to overfit than I expected. Even 'good fits' don't fit as closely as I expected
- I expected the function which aimed to fit the pdf of the Gaussian mixture only in the places where there were samples to estimate a function which looked like the pdf (ie predicted ~0 where there was no data). However, this is not the case because it was not given 'negative' examples (samples where pdf ~0). This means that any estimate is OK outside the range of data because it does not effect the cost function (although it does matter for generalisation)
    - The network's job is to approximate the function for the given in,out training data. outside the range of data, the values do not matter in terms of the cost function (but do in terms of generalisation).
    - If more contrast is needed between the areas with/without data, 'negative samples' could be added in the places without data to make the cost function consider those areas.
- For the task: the important thing is that in the areas without data, the uncertainty of the result is high, the actual predicted value does not matter as much, however roughly interpolating between the areas with data would be sensible.
- cross entropy for binary classification: `-1/n*sum(y_hat*log y + (1-y_hat)*log (1-y))`
    - when there are $k$ classes and $N$ training examples: $-\frac 1 n\sum_n^N\sum_k^K \hat Y^{(n)}_k\log_2 Y^{{n)}_k$

# Things I learned about tensorflow
- output layers can be sliced and different activation functions can be applied to different sections of it, eg to output several outputs with softmax, some more with linear etc


# Tips for tuning hyper-parameters
- first list out the hyperparameters to tune and potentially rule out the ones that will be kept fixed (eg like the layer activation functions)
