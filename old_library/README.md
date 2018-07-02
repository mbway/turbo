# Overview of the source code #

## optimisation.py
contains the definitions of `Job`, `Sample`, `Evaluator`, `Optimiser`, `GridSearchOptimiser`, `RandomSearchOptimiser` and `BayesianSearchOptimiser`

## optimisation_net.py
contains networking utilities for the evaluator/optimiser client/server

## optimisation_gui.py
contains utilities for opening Qt GUI interfaces for the client/server. Also has utility functions for use with Jupyter notebooks

## optimisation_utils.py
small utilities that don't belong anywhere else

## optimisation_test.py
contains all the automated unit and system tests

## synthetic_data.py
several utilities for creating synthetic data for testing the optimisation library

## plot3D
a wrapper around plotly for creating 3D surface and scatter plots

