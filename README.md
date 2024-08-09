# Hillslope project
This project contains the code used to generate and analyse a machine learning based minimal model of a semi-arid hillslope ecosystem.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation and usage](#installation-and-usage)
- [Contact Information](#contact-information)

## Project Overview

In this project, a Random Forest and a Neural Network are used to simulate the aggregate behaviour of vegetation and soil, using data from either a minimal differential equation model, or a spatially detailed simulation model. This enables an evidence-based analysis of the ecosystem behaciour. More information in the following EGU abstract: https://doi.org/10.5194/egusphere-egu24-11880.

## Installation and usage

The scripts are all ran in python, so a conda environment needs to be created from the ```env.yml``` file provided. The detailed model data (if relevant) should be added to the ```data/raw``` folder, and the simulations used for evaluating the models in forward simulation to the ```data/raw/fwd_sim``` folder.

To configure the script, one can do so in the ```src/config.py``` file. A brief explanation of each of the parameters included can be found within it. Once the configuration values are set, it is only needed to run the ```src/main.py``` script. The outputs will be stored in each of the subfolders inside the ```results``` folder.

## Contact Information

For any questions regarding the code or this project, please contact o.pomarolmoya@uu.nl.