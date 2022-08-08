# Quantum Topological Data Analysis

This is a `Python` library for quantum topological data analysis.
It can be used to extract persistent Betti numbers from point clouds and time series data sets.
It can also use the persistent Betti numbers to construct a persistence diagram.

## Dependencies

You must have `python 3.9` installed in your system.
You can use `pyenv` to manage different python versions in your system.

The project includes a `Pipfile` that can be used to create a virtual environment with all the necessary packages.
In order to do this you must have `pipenv` installed in your system.

## Installation

Clone the git repository to the directory where you want to install it.

	git clone https://github.com/sandman-ba/takens.git path/to/project/directory
	
Or if you prefer to connect using SSH:

	git clone git@github.com:sandman-ba/takens.git path/to/project/directory
	
Move to the directory where you cloned the repo and use `pipenv` to create a virtual environment for the project and install all the necessary packages.

	cd path/to/project/directory
	pipenv install
