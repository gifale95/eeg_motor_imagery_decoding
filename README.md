# Project aim
This project aims at creating an EEG [encoding][enc] model of visual object perception and recognition. Once trained (through a linear regression), the model should be able to predict the EEG responses to a given image using a DNN's internal representations (activations) of that very same image.

[enc]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3037423/



# Stimuli images
All images are squared and represent objects on a natural background. The stimuli set is divided into a **training** partition, a **validation** partition, a **test** partition and a **target** partition.
- The **training** partition consists of 7500 image conditions (1500 object categories, with 5 exemplar for each category) coming from the [THINGS database][things].
- The **validation** partition consists of 50 image conditions (50 object categories, with 1 exemplar for each category) coming from the [THINGS database][things].
- The **test** partition consists of 50 image conditions (50 object categories, with 1 exemplar for each category) coming from the [ImageNet database][imagenet].
- The **target** partition consists of 10 images representing the fictional characted *Buzz Lightyear* from the animated cartoon *Toy Story*.
The stimuli images can be found in the Curta directory: "*/home/giffordale95/studies/eeg_encoding/paradigm_2/image_set*".


[things]: https://www.biorxiv.org/content/10.1101/545954v1
[imagenet]: http://image-net.org/papers/imagenet_cvpr09.pdf



# Experimental design
The experiment consists in a rapid-image-presentation paradigm. Rapid sequences of 20 images are presented to participants in a time frame of 4s, where each image stays on screen for 100ms, followed by an interstimulus interval of 100ms. At the end of each sequence the participants are asked to blink their eyes, and to report (with a key press) whether the target image (*Buzz Lightyear*) was present or not in the seqeunce. 53 sequences add up to one run, and one data collection session consists of 19 runs: runs 1-3 containing **test** images, run 4 containing **validation** images, and runs 5-19 containing **training** images.



# EEG and behavioral data
Each of the 7 participants underwent 1, 2 or 3 data collection sessions, which resulted in raw EEG and behavioral data:
- **Subject 1:** 2 sessions;
- **Subject 2:** 1 session; 
- **Subject 3:** 1 session; 
- **Subject 4:** 1 session; 
- **Subject 5:** 3 sessions; 
- **Subject 6:** 2 sessions; 
- **Subject 7:** 3 sessions. 
The EEG data was recorded using a 64 channels cap, with a sampling rate of 1000Hz.
Each data collection session resulted in 2 repetitions of the 7500 **training** image conditions, 20 repetitions of the 50 **validation** image conditions, and 60 repetitions of the 50 **test** image conditions.
In order to run the analyses code, the EEG and behavioral data should have the same sorting format found in the curta directory: "_/scratch/giffordale95/studies/eeg_encoding/paradigm_2/dataset_".



# DNN activation data
The DNN data consists in the DNN activations of the stimuli images. The current project uses a ResNet50 architecture ([He et al., 2015][resnet]). The resulting activations are stacked together across blocks/layers, Z-scored and downsampled to different principal components numbers using Principal Component Analysis (PCA).
In order to run the analyses code, the DNN data should have the same sorting format found in the curta directory: "_/scratch/giffordale95/studies/eeg_encoding/paradigm_2/dnn_activations/pca_activations_blocks_combined_".

[resnet]: https://arxiv.org/abs/1512.03385



# Code
The */code* folder contains:
- **"_001-data_collection_":** the *.m* scripts  used for data collection;
- **"_002-preprocessing_":** the *.py* and corresponding *.sh* scripts to preprocess the EEG data either sequentially on a local computer or in parallel on Curta;
- **"_003-data_analysis_":** the *.py* and corresponding *.sh* scripts to run the analyses on the EEG data either sequentially on a local computer or in parallel on Curta;
- **"_004-plotting_":** the *.py* scripts used for plotting the analyses results.
For the code to work, the EEG and DNN data should be placed in the same directory (the project directory). This is also the directory in which the analysis results will be stored.