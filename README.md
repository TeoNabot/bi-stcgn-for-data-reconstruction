**How to use**
This project includes a main.ipynb notebook (inside the notebooks directory), from which the project can be executed. 
To run the notebook, you have to separately download the data.txt file. Then, to tell the main notebook where to look, the config.yaml file needs to be updated. Specifically, the raw_data path needs to point to the directory where the data.txt file is stored. By default it points to data/raw. The paths are relative to the project root. I included the sensor coordinates files for simplicity inside the repository. 
I also included a requirements.txt file, indicating the packages that should be installed before running the code.
Finally, to avoid retraining the model, the model weights can be downloaded. Then, the config.yaml file can be updated in order to point to the desired path. Specifically, the model .pth file should be stored under the "checkpoints" path, with the name "bistgcn_best.pth".
