# Bi-Directional Spatio-Temporal Graph for Data Reconstruction

This repository implements a Bi-Directional Spatio-Temporal Graph Neural Network designed for data reconstruction in the Intel Berkley Research Lab Dataset. By capturing both spatial dependencies between graph nodes and the bidirectional temporal dynamics—leveraging both past and future contexts—the model recovers structurally missing data points.

## Key Features

* **Bi-Directional Temporal Modeling:** Utilizes forward and backward temporal information to provide richer context for accurate data imputation.
* **Spatial Dependency Extraction:** Models spatial structure across the network to improve data reconstruction.

## How to use
This project includes a main.ipynb notebook (inside the notebooks directory), from which the project can be executed. 
To run the notebook, you have to separately download the data.txt file. Then, to tell the main notebook where to look, the config.yaml file needs to be updated. Specifically, the raw_data path needs to point to the directory where the data.txt file is stored. By default it points to data/raw. The paths are relative to the project root. I included the sensor coordinates files for simplicity inside the repository. The data.txt file can be found at https://www.kaggle.com/datasets/divyansh22/intel-berkeley-research-lab-sensor-data. 
I also included a requirements.txt file, indicating the packages that should be installed before running the code.

