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

	git clone https://github.com/sandman-ba/QTDA.git path/to/project/directory
	
Move to the directory where you cloned the repo and use `pipenv` to create a virtual environment for the project and install all the necessary packages.

	cd path/to/project/directory
	pipenv install

## References

See the [Quantum Persistent Homology](https://link.springer.com/article/10.1007/s41468-023-00160-7 "Quantum Persistent Homology") paper for details regarding the quantum algorithm for persistent homology:

> Ameneyro, Bernardo, Vasileios Maroulas, and George Siopsis. "Quantum persistent homology." *Journal of Applied and Computational Topology* 8, no. 7 (2024): 1961-1980.

See the [Quantum Persistent Homology for Time Series](https://ieeexplore.ieee.org/abstract/document/9996768 "Quantum Persistent Homology for Time Series") paper for details about the quantum oracle for time series.

> Ameneyro, Bernardo, George Siopsis, and Vasileios Maroulas. "Quantum persistent homology for time series." In *2022 IEEE/ACM 7th Symposium on Edge Computing (SEC)*, pp. 387-392. IEEE, 2022.
