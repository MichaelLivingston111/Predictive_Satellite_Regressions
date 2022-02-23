# Remote sensor algorithms: Predicting marine exopolymer concentrations over space and time


Marine exopolymers are an important element in global carbon cycle as they are critical for the sequestraion of carbon dioxide in the ocean. Furthermore, they are an important substance in the desalination industry as they frequently clog reverse osmosis filters critical for efficient desalination. As such, it is important to develop models that may accurately predict marine expolymer concetrations. 

This project creates a series of predictive models to estimate the concentrations of exopolymers at the oceans surface from a suite of commomly reported variables on both remote sensors and satellites. 

The "model selection" file creates and validates a series of models. Three more files specify the exact details for each independent model, named "Satellite_sensor_model", "Remote_sensor model", and "Feature_model" with details therein. The "model prediction" and "model prediction function" file derives the actual predictions from data files, while the "model visualization" file visualizes the results from a few select periods.

Overall, the models are quite accurate when tested on validation data from marine surfaces with an accuracy of >81% and mean error of ~10 micrograms. The models do not appear to be overfit, or violate any necessary assumptions.

This project is a completely novel approach to estimating concentrations of marine exopolymers. Large scale estimations have never been done before. Furthermore, this represents a valuable tool for desalination industries as marine exopolymer gels are a serious issue in the desalination industry.  
