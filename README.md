# Predictive_Satellite_Regressions

A project that utilizes satellite data from NASA and predictive regressions on a global scale.

Marine exopolymers, or 'gels' are an important substance for both the global carbon cycle and the desalination industry. These polymers are critical for the sequestraion of carbon dioxide in the ocean. Furthermore, they are well known to frequently clog reverse osmosis filters critical for efficient desalination.

This project creates a predictive model to estimate the concentrations of these exopolymers at the oceans surface using chlorophyll and temperature from satellite data as predictor variables. The model uses both of these variables in a mixed effects model, with 'season' as a random intercept.

The model satisfies all assumptions of a linear model. Model 5X cross validation was very promising, validating the use of this model. The model was hten applied to satellite data from the years 2012 - 2021 (only one year shown in this repository as an example).

Here, there is code files from both python and R detailing algorithms for linear mixed effects models. The python algorithm will be used in to predict values from the satellites, but the R algorithm can be visualized and validated better. 
