
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

# Things I learned about Gaussian processes
- they are not translation invariant. The prior assumes a mean of 0, so this dictates how the function will tend when extrapolating. Take a small data set and add a constant to the y and observe how the GP prediction changes.

# Things I learned about Bayesian optimisation
- care has to be taken regarding the surrogate function or the results could be worse than random
- Gaussian processes are harder to fit to data than expected
- Plotting the results can help determine if the parameter ranges you imposed are too restrictive as the optimiser always wants to sample at the extreme of a parameter
- with EI and PI acquisition functions it is important to find the global maximum of the acquisition function because it is often a very thin spike and so easy to miss during optimisation. UCB does not suffer from this as much.
- favouring exploration seems to generally perform better, at least more consistently
- UCB appears to be the best acquisition function for most of my tests
- choosing randomly instead of sampling very close to an existing sample does help, but choosing a tolerance too large does more harm than good. For some functions, I got better results from having a very small (1e-8) tolerance to allow the Bayesian optimisation to have more control over where to sample.
    - for very smooth functions (such as the toy ones I used during development) a large close tolerance is fine.
- parallel Bayesian optimisation improves performance, however if it causes the GP fit to cause an inaccurate fit then it may be harmful (perhaps no worse than random). Explicitly accounting for noise in the GP kernel may help with this.
- choosing an objective function is important and very challenging
- discrete parameters can be simulated by rounding the given continuous value.
- Bayesian optimisation can handle pretty well with a noisy objective function, the GP will smooth out the surrogate function allowing good samples to still be chosen.
- unfortunately, different objective functions can be fitted by different GP kernels, however without knowing the true function it is hard to determine the correct kernel to use.

# Things I learned about GPs
- standardizing the data (ie subtract mean, divide by variance) is _essential_. I failed to fit a GP when I had a dataset which was fairly reasonable and I gave the GP many samples to fit to with many optimiser repeats and I also narrowed the boundary on the kernel hyperparameters to prevent testing obviously bad values and to speed up the optimisation (which I think it did).
    - I was getting nowhere, then the first try after standardizing the data and the GP produced reasonable results
- when the GP does not fit at all and describes the variation in the data as being completely noise, this may indicate that more restarts of the GP optimiser is required
- if you know what the surrogate function should look like then you can set the surrogate GP parameters to fixed values (eg the length scale should be the average distance between the points)
- restrict the bounds of the WhiteKernel used for the surrogate GP to prevent it from explaining all variation as noise. May be useful to plot the noise levels throughout the optimisation to get an idea for what maximum noise level might be appropriate for the function. As a Heuristic: when normally distributed noise sigma=0.3 is added, a maximum noise level of 0.15 seems reasonable (according to one of the tested functions).
```
xs = [s.gp.kernel_.k2.noise_level for n, s in optimiser.step_log.items()]
plt.plot(xs)
```
- Bayesian optimisation likes to test points around the edges, which can be problematic. For example testing out a neural network with max width, max depth and max epochs is not a good idea. Using an acquisition function which takes the evaluation time into account may help prevent this.
- one major problem is when the surrogate function explains the data as being complete noise with a straight line mean. This is problematic because there is typically a slight gradient in the mean prediction line which results in a 'corner' of the configuration space to be sampled repeatedly. Usually after taking more samples the GP once again fits the data. Increasing the number of GP optimizer restarts helps. Unsurprisingly, the quality of the surrogate model dramatically changes the performance of the algorithm because it relies on the surrogate being accurate in order to make informed decisions. If the surrogate is misleading then Bayesian optimisation is useless (probably worse than random).

# Things I learned about networking
- it is so much better to extract all the networking logic and abstract it when interacting with the rest of the program logic. It makes it much easier to analyse the behaviour of the protocol by reducing the number of possible execution paths
- sockets sometimes break when you don't expect them to, and sometimes don't break when you expect them to. So always handle every possible failure when networking is involved
- TCP is not as reliable as I thought, the checksum is weak and other errors which I simulated broke through to the application layer where I had to deal with them manually
- having tried both approaches, I think that it is better to have more short lived connections rather than keeping a few connections open for a long time. It makes re-starting the connection easier when an error occurs and a thread pool is not required to deal with all the open connections, so the server can be serialised (which may actually increase performance)
    - single-threaded logic is also much simpler as queues and locks are not required
- when designing something to be reliable, it seems to be a good idea to have the last message be some confirmation from the peer expecting some important data, if the sender of the important data receives the confirmation then they can be quite sure that everything is fine.
- it helps to draw out the protocol through every possible trace, including all possible errors and checking how your code would respond

