# Violence-Detction-3DCNN
# Violence-Detection-using-3D-CNN
## Preparation of Dataset
##### Collect all the Violence and Non Violence videos in a folder for training.
##### Make the videos of larger length to videos of same length either manually splitting or by running the below python file
##### [`Clipping.py`](/Clipping.py)
##### The files in the dataset can also renamed by executing the below file
##### [`Renaming_files.py`](/Renaming_files.py)
## Download already available dataset
##### Click on the links below to download the complete dataset and also the small test dataset
##### Complete Dataset:
##### https://drive.google.com/drive/folders/1D_eP8SwJptLxzoGKIZxGrjF6GNMDT0zc?usp=sharing
##### Test Dataset:
##### https://drive.google.com/drive/folders/1uhlzSc_QCCdNCrp7PSHyi32mf0XDJE-f?usp=sharing
## Extract the features for training
##### Run the python file below to extract the features from the videos
##### [`Feature_Extraction.py`](/Feature_Extraction.py)
##### After feature extraction the features are stored in .npy files which can later be used for training
## Training 3D-CNN Model
##### Run the python file below with the existing feature files to train the model
##### To train the model for different variations of Batch size and number of filters run the python file below
##### [`Training_Multiple.py`](/Training_Multiple.py)
##### The training is done for different Batch_sizes and number of filters, after completion of each training process the weights are stored in the specified folder. A text file is also created containing the accuracy values for all the trained models
##### To train the model only once with particular parameters run the python file below
##### [`Training.py`](/Training.py)
##### After the completion of training the weights are stored in the specified folder
## Obtaining Predictions
##### After the training process the existing weights with better accuracy can be used for making the predictions on the new set of videos
##### The small test dataset downloaded can also be used for making the predictions
##### Run the below python file to get the predictions
##### Predictions for different multiple weight files can be obtained by running the python flie below
##### [`Predictions_Multiple.py`](/Predictions_Multiple.py)
##### The predictions are stored in a text file 
##### Predictions for single weight file can be obtained by executing the python file below
##### [`Predictions.py`](/Predictions.py)
## References
##### [1] Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning Spatiotemporal Features with 3D Convolutional Networks. 2015 IEEE International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv.2015.510
