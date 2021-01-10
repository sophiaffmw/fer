# VGG-19 with attention for Facial Expression Recognition
A VGG-like model on facial expression recognition (FER2013)

## Dependencies ##
- Python 2.7
- Pytorch >=0.2.0
- h5py (Preprocessing)
- sklearn (plot confusion matrix)


## FER2013 Dataset ##


### Preprocessing Fer2013 ###
- first download the dataset(fer2013.csv) then put it in the "data" folder, then
- python preprocess_fer2013.py

### Train and Eval model ###
- python3 main.py --model VGG19 --bs 128 --lr 0.01
