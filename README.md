# Bi-Directional Spatio-Temporal Graph for Data Reconstruction

This repository implements a Bi-Directional Spatio-Temporal Graph Neural Network designed for robust data reconstruction. By capturing both the non-Euclidean spatial dependencies between graph nodes and the bidirectional temporal dynamics—leveraging both past and future contexts—the model effectively recovers missing, sparse, or corrupted data points within complex network structures.

## Key Features

* **Bi-Directional Temporal Modeling:** Utilizes forward and backward temporal information to provide richer context for accurate data imputation.
* **Spatial Dependency Extraction:** Models complex topological structures across the network to understand how neighboring nodes influence one another.
* **Robust Reconstruction:** Designed to handle discontinuous or sparse inputs to reliably estimate unobserved states.
* **Streamlined Training Pipeline:** Built in Python with a clean, straightforward train/validation split for efficient model evaluation without unnecessary computational overhead.

## How to use
This project includes a main.ipynb notebook (inside the notebooks directory), from which the project can be executed. 
To run the notebook, you have to separately download the data.txt file. Then, to tell the main notebook where to look, the config.yaml file needs to be updated. Specifically, the raw_data path needs to point to the directory where the data.txt file is stored. By default it points to data/raw. The paths are relative to the project root. I included the sensor coordinates files for simplicity inside the repository. 
I also included a requirements.txt file, indicating the packages that should be installed before running the code.

