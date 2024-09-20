# Requirements and instructions to run the framework

Before delving into the details of the project files, please consider that this project has been executed on a Windows 10 machine with Python 3.11.5. There are a few libraries that have been used within Python modules. Among these, there are:

- tensorflow 2.13.0
- scipy 1.11.2
- scikit-learn 1.3.0
- pm4py 2.2.20.1

Please note that the list above is not comprehensive and there could be other requirements for running the project.

The framework runs by executing the DOS experimentation.bat script. This script includes a set of parameters that can be customized to set: the number of repetitions with which a technique of the framework is executed; the datasets to which the techniques are applied; the number of traces per event log; the test percentage; the validation percentage; the process mining-based feature extraction approaches; the dimensionality reduction techniques; and the baseline techniques. The default values reflect the experimental settings used for the paper related to this project. 

The experimentation.bat script cleans the environment and, for each framework technique, combines the execution of the preprocessing.py, pm_fe.py and dr_ad.py Python scripts, which, respectively, split the event data of the selected datasets according to the number of traces per event log, apply to the event logs the selected process mining-based feature extraction technique, and perform anomaly detection by the selected dimensionality reduction technique. The results are collected under the Results folder.
