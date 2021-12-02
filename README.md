# Predictive_Satellite_Regressions

A project that utilizes satellite data from NASA and predictive models to estimate the concentrations of important carbon-based exopolymers on a global scale.

Marine exopolymers, or 'gels' are an important substance for both the global carbon cycle and the desalination industry. These polymers are critical for the sequestraion of carbon dioxide in the ocean. Furthermore, they are well known to frequently clog reverse osmosis filters critical for efficient desalination. Therefore, it is important to accurately estimate their concentrations across a vareity of marine environments. 

This project creates a predictive model to estimate the concentrations of these exopolymers at the oceans surface using chlorophyll and temperature from satellite data as predictor variables. The model uses both of these variables in a linear mixed effects model, with 'season' as a random intercept. This model is created and applied in the Rstudio "linear mixed effects model file". This model was selected from a variety of other models based off of cross validation and test results. See the model cross validation, bootstrapping and selection files for more details.

The model satisfies all assumptions of a linear model. Model 5X cross validation was very promising, validating the use of this model. The model was then applied to satellite data from the years 2012 - 2021 (only one year shown in this repository as an example thus far).

Here, there is code files from both python and R detailing algorithms for linear mixed effects models. The python algorithm is be used to upload, index and sort the satellite data, as well as plot all of the predictions. Rstudio is used to create and validate the model, as well as make all the predictions.
