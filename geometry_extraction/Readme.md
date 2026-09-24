# Application of CNN on Crystal Plasticity for Material Modeling
Developed a CNN Model to implement Crystal Plasticity Modeling
and predict the behaviour of materials without the need of expensive simulations.

## Table of contents

- Requirements
- Installation
- Run

## Requirements

This module requires the following modules:

* [damask==3.0.0a7.post0](https://pypi.org/project/damask/)
* [h5py==3.7.0](https://pypi.org/project/h5py/)
* [matplotlib==3.6.3](https://matplotlib.org/3.6.3/)
* [numpy==1.23.4](https://pypi.org/project/numpy/1.23.4/)
* [scikit-learn==1.2.0](https://pypi.org/project/scikit-learn/1.2.0/)
* [scipy==1.9.3](https://pypi.org/project/SciPy/1.9.3/)
* [torch==2.1.2+cu118](https://pytorch.org/get-started/locally/)
* [torchmetrics==0.11.4](https://pypi.org/project/torchmetrics/0.11.4/)
* [torchvision==0.16.2+cu118](https://pypi.org/project/torchvision/) 


## Installation

* Install Python 3.10.
* Open Windows command prompt.
* Change the directory by – “cd” command followed by the path where you want to create the virtual environment.
* Create a virtual environment named venv by command "python -m venv venv".
* Activate the virtual environment using command "venv\Scripts\activate"
* Install all the modules mentioned in Requirements.

## Run

* Extract the "Data" folder to the desired directory and transfer "50 grain" folder outside the "Data" folder.
* In the "data_extraction.py" file, change line 34 to "path_file = os.path.join(os.getcwd(), "50 grains", "*.hdf5")".
* The "extracted_data.h5" is already provided in folder. However, running the "data_extraction.py" produces "extracted_data.h5" file. 
* Ensure "extracted_data.h5" is saved in a folder named "Data".
* Run "model.py". The Model layers can be tweaked in the class Custom3dCNN() by addition or removal of layers in both "__init__" and "forward()". Preferrably comment the layers you do not want in the model.
* On running the file, a CNN model is created which is trained on the dataset obtained from "extracted_data.h5". 
* The model is saved at the end in a folder named "Model" and the learning curves and metrics are saved in the current directory as images.

## Troubleshoot
* Run "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118" if there is any issue in installing torchvision.
