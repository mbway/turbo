# README #

# Contents (in chronological order) #
First I experimented with tensorflow and Jupyter, then a simple MC-dropout implementation
- Linear Regression.ipynb
    - following a tensorflow tutorial to create a simple model (linear regression)
- MLP.ipynb
    - experiment with a simple MLP
- Animation.ipynb
    - experiment with animations in Jupyter
- MLP-Dropout-Uncertainty.ipynb
    - an ad-hoc (no external modules) implementation of MC-dropout with a simple MLP with 1D data (complete with equations and explanations)

I then tried to generate synthetic 2D data (Gaussian mixture pdf) which was much harder than expected. I then implemented MC-dropout with the 2D data
- Gaussian-Implementation.ipynb
    - experiment with my own multivariate Gaussian PDF
- MLP-Dropout-Uncertainty-2D.ipynb
    - MC-dropout being used to estimate uncertainty in 2D (ad-hoc, no external modules)
- MLP-Dropout-Uncertainty-2D-ReLu-params.ipynb
    - a copy of `MLP-Dropout-Uncertainty-2D.ipynb` with relu activation and some fairly decent parameters for it

Preparing for automatic hyperparameter optimisation, I extracted the important code into python modules:
- MLP.py
    - A simple multi-layer perceptron implementation
- synthetic_data.py
    - functions for creating the synthetic data first created in `MLP-Dropout-Uncertainty.ipynb` and `MLP-Dropout-Uncertainty-2D.ipynb`
- MLP-module.ipynb
    - a re-creation of `MLP-Dropout-Uncertainty.ipynb` but using the two modules above

I then started an optimisation framework and interactive 3D plotting tools
- plot3D.py
    - a small wrapper around some plotly functions to easily create 3D scatter and surface plots
- 3D-Plotting.ipynb
    - a demonstration of using plot3D.py and a comparison with a (non-interactive) matplotlib 3D plot
- optimisation.py
    - a self-contained optimisation framework capable of Grid search, random search and Bayesian optimisation. Other features: tested, Very general and flexible, multithreaded, built in plotting tools, good logging and error handling, can save (to json) and resume optimisation
- optimisation_tests.py
    - unit tests and system tests for the optimisation framework
- Optimisation.ipynb
    - A showcase of the features of optimisation.py with some toy problems in 1D and 2D
- MLP-Optimisation.ipynb
    - application of the optimisation framework to optimise the MLP hyperparameters to give the best uncertainty predictions


# Notes #
- a neural network with infinite neurons is equivalent to a Gaussian process
- can use dropout to approximate the uncertainty of the network prediction


## Sources ##
- https://github.com/aymericdamien/TensorFlow-Examples
- 'estimating uncertainty using dropout'
- http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html
- https://github.com/yaringal/DropoutUncertaintyCaffeModels
- Gaussian process
    - http://www.gaussianprocess.org/gpml/chapters/

