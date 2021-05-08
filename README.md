# Guided-EMO-SpatialTransformer
Repository to IMDB publication

    <FER2013-dir>
    <AffectNet-dir> -> Directory with the images of AffectNet (Manually_Annotated_Images folder)

## Installation
Download and install the following GitHub repository: 

    https://github.com/alexanderkroner/saliency


and include the files of the respository into: RAVDESS_DS/src/VIDEO/saliency/Saliencia
After downloading the files, create a new virtualenv and install the packages of this repository.

To install the other packages, create a new virtual environment and run:

    * pip install -r requirements.txt

** If problems installing grpcio, update pip version: pip3 install --upgrade pip

## Prepare data

First, we need to prepare the Datasets. 
For Affectnet, we apply a change in the format of some images (.bom, .tiff...) since the saliency extractor do not accept these formats. To do so, run: 

    python3 src/DSpreparation/prepare_AffectNet_imgs.py -r <AffectNet-dir> -o <output-dir-with-grayscaled-48x48-imgs>

### Extract landmarks: 
To extract landmarks, run:  

    python3 src/ImagePreprocessing/LandmarkExtractor.py -o <AffectNet-dir>/AFFECTNET/LANDMARKS_dlib_MTCNN -olandm <AffectNet-dir>/AFFECTNET/LANDMARKS_dlib_MTCNN_npy
    -d <AffectNet-dir>/AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images -trainCSV <AffectNet-dir>/AFFECTNET/datasetFinal.csv 
    -trainCSVSep , -ds AffectNet -logs <AffectNet-dir>/AFFECTNET

    ## FER=2013:


    -o /mnt/RESOURCES/FER2013/LANDMARKS_dlib_MTCNN_PRUEBA -olandm
    /mnt/RESOURCES/AFFECTNET/LANDMARKS_dlib_MTCNN_npy_PRUEBA
    -d
    /mnt/RESOURCES/AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images
    -trainCSV
    /mnt/RESOURCES/AFFECTNET/datasetFinal.csv
    -trainCSVSep
    ,
    -ds
    AffectNet
    -logs
    /mnt/RESOURCES/AFFECTNET



### Extract saliency:
1. Change to the virtualenv where you installed the packages for extracting saliency
2. Run the following command:
   

    python3 python3 main.py test -d salicon -p ../AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images/1 -o ../AFFECTNET/SALIENCY -Ds RAVDESS
   
    python3 Saliencia/main.py test -d salicon -p ../DATASETS/FER2013/fer2013/IMAGES -o ../DATASETS/FER2013/fer2013/SALIENCY -Ds FER2013

    * where PATH is the directory with the input images 
    * and OUT_PATH is the directory where you want to save the saliency of the images
    ** IMPORTANT: The original code does not include the -o option, check initial main in this repository to include this option

## Train models:
In this section, we summarize the parameters selected to train the models in their different versions. 

### Original models + Original img through spatial attention layer
To train the original models (baseline version of 48x48 from: ) and implemented version for 100x100 size images, with a 5-folds cross-validation 
strategy and replicate our way to train the algorithms, run the following code:

    python3 src/main_CV_Pytorch -kf 5 
            -d ../RAVDESS/AV_Speech_labels_polarityCOMPLETE_numeric.csv 
            -r ../RAVDESS/AV_VIDEOS_FRAMES/frames -imgSize 48 -e 50
            -lr 0.001 -bs 32 -s 2020 -logs ../RAVDESS_DS/data/RAVDES_logs
      


### Porblems with images:
    103/29a31ebf1567693f4644c8ba3476ca9a72ee07fe67a5860d98707a0a.jpg -> Not included