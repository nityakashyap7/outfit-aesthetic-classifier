# Outfit Aesthetic Classifier
#### USC Spring 2025 CSCI 467 course final project using SVMs and CNNs trained on a custom scraped dataset

## Contributors
- [Maia Nkonabang](https://github.com/maiankona)
- [Nitya Kashyap](https://github.com/nityakashyap7)


## Abstract
This project uses machine learning to classify outfit images into aesthetic categories (e.g., Y2K, academia, punk). We scraped over 14,000 Pinterest images via Google Images, preprocessed and labeled them across 10 style categories, and tested multiple classification approaches. Our baseline method used multi-class Support Vector Machines (SVMs) with handcrafted features such as color histograms and Local Binary Patterns (LBP). We trained a convolutional neural network (CNN) from scratch using PyTorch for our main models. Also, we experimented with transfer learning via a pre-trained ResNet backbone connected to an MLP head. Evaluation across training, development, and test splits shows that the pretrained CNN outperforms the SVM baselines and the basic CNN in accuracy and generalization. We also identified key challenges around overlapping aesthetics and lack of computer resources. 

Read the full report [here](https://github.com/nityakashyap7/outfit-aesthetic-classifier/blob/main/CSCI_467_Project_Final_Report.pdf).  
All code used in this project is publicly available in this repository.   
Check out the dataset [here](https://doi.org/10.5281/zenodo.15164901).   

## Commands to reproduce results
`python3 baseline.py` Runs SVM baseline  
`python3 cnn.py cnn` Runs standard CNN  
`python3 pretrained plus mlp.py` Runs pretrained model with MLP  

