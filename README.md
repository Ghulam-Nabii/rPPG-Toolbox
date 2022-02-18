
# Overview

xxx is a python toolbox aiming for rPPG signal extraction supporting bothe deep-learning
and signal processing methods. #TODO: Adds more description.

# Requirments

# Dataset
The toolbox supports three datasets, which are UBFC, PURE and COHFACE.

You should cite corresponding papers when using them.

* [UBFC](https://sites.google.com/view/ybenezeth/ubfcrppg)
    * S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
    * In order to use this dataset in a deep model, you should organize the files as follows:
    -----------------
         data/UBFC/
         |   |-- subject1/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |   |-- subject2/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |...
         |   |-- subjectn/
         |       |-- vid.avi
         |       |-- ground_truth.txt
    -----------------
* [COHFACE](https://www.idiap.ch/en/dataset/cohface)
    * Guillaume Heusch, André Anjos, Sébastien Marcel, “A reproducible study on remote heart rate measurement”, arXiv, 2016.
    * In order to use this dataset in a deep model, you should organize the files as follows:
    -----------------
         data/COHFACE/
         |   |-- 1/
         |      |-- 0/
         |          |-- data.avi
         |          |-- data.hdf5
         |      |...
         |      |-- 3/
         |          |-- data.avi
         |          |-- data.hdf5
         |...
         |   |-- n/
         |      |-- 0/
         |          |-- data.avi
         |          |-- data.hdf5
         |      |...
         |      |-- 3/
         |          |-- data.avi
         |          |-- data.hdf5
    -----------------
     
* [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
    * Stricker, R., Müller, S., Gross, H.-M.Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
    * In order to use this dataset in a deep model, you should organize the files as follows:
    
    -----------------
        data/PURE/
         |   |-- 01-01/
         |      |-- 01-01/
         |      |-- 01-01.json
         |   |-- 01-02/
         |      |-- 01-02/
         |      |-- 01-02.json
         |...
         |   |-- ii-jj/
         |      |-- ii-jj/
         |      |-- ii-jj.json
    -----------------
# Deep Model
# Configuration
TODO
# Run
## Deep Model Training

python main.py --config_file  *[path for the config file]* --data_path *[path for dataset]*  --preprocess

You need to keep the preprocess arg unless this is not the first time you run the same pipeline and want to skip data preprocessing.

Here are some exemples. The data should be organized as mentioned.

* Train **physnet** on **COHFACE** dataset using **cuda:0**
    * python main_neural_method.py --config_file configs/COHFACE_PHYSNET_BASIC.yaml --data_path [YOUR DATA PATH] --preprocess 
* Train **tscan** on **COHFACE** dataset using **cuda:0**
    * python main_neural_method.py --config_file configs/COHFACE_TSCAN_BASIC.yaml --data_path [YOUR DATA PATH] --preprocess 
* Train **physnet** on **PURE** dataset using **cuda:0**
    * python main_neural_method.py --config_file configs/PURE_PHYSNET_BASIC.yaml --data_path [YOUR DATA PATH] --preprocess 
* Train **tscan** on **PURE** dataset using **cuda:0**
    * python main_neural_method.py --config_file configs/PURE_TSCAN_BASIC.yaml --data_path [YOUR DATA PATH] --preprocess 

## Evaluation

* Test **physnet** on **COHFACE** dataset using **cuda:0**
    * python evaluation_neural_method.py --config_file configs/COHFACE_PHYSNET_EVALUATION.yaml --data_path [YOUR DATA PATH] --preprocess 
* Test **tscan** on **COHFACE** dataset using **cuda:0**
    * python evaluation_neural_method.py --config_file configs/COHFACE_TSCAN_EVALUATION.yaml --data_path [YOUR DATA PATH] --preprocess 
* Test **physnet** on **PURE** dataset using **cuda:0**
    * python evaluation_neural_methods.py --config_file configs/PURE_PHYSNET_EVALUATION.yaml --data_path [YOUR DATA PATH] --preprocess 
* Test **tscan** on **PURE** dataset using **cuda:0**
    * python evaluation_neural_methods.py --config_file configs/PURE_TSCAN_EVALUATION.yaml --data_path [YOUR DATA PATH] --preprocess 


## Tensorboard
command: tensorboard --logdir=runs/


