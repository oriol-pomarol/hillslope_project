# Hillslope project
This project contains the code used to exemplify the data driven discovery of the time transfer function of a semi-arid hillslope ecosystem.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation and usage](#installation-and-usage)
- [Contact Information](#contact-information)

## Project Overview

In this project, a Random Forest and a Neural Network are used to simulate the aggregate behaviour of vegetation and soil, using data from either a minimal differential equation model, or a spatially detailed simulation model. This enables an evidence-based analysis of the desertification mechanisms. More information in the following EGU abstract: https://doi.org/10.5194/egusphere-egu24-11880.

## Installation and usage

The scripts are all ran in python, so a conda environment will need to be created from the ```env.yml``` file provided. If needed to run on detailed model data, the training data (accessible at https://doi.org/10.5281/zenodo.13384361) should be added to the ```data/raw``` folder. Anything that should be used for testing the models in forward simulation should be added to the ```data/raw/fwd_sim``` folder. 

To configure what and how to run it, one can do so in the ```src/config.py```. An explanation of each of the parameters can be found inside that file. Once the configuration values are set, it is only needed to run the ```src/main.py``` script. The outputs will be stored in each of the subfolders inside the ```results``` folder.

## Contact Information

For any questions regarding the code or this project, please contact o.pomarolmoya@uu.nl.
