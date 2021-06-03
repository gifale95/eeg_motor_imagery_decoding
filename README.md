# Decoding Motor Imagery states from EEG data using Convolutional Neural Networks
This repository provides Python code for the decoding of different motor imagery conditions from raw EEG data, using a CNN.



## Environment setup
To run the code, create and activate a dedicated Anaconda environment by typing the following into your terminal:
```shell
curl -O https://raw.githubusercontent.com/gifale95/eeg_motor_imagery_decoding/main/environment.yml
conda env create -f environment.yml
conda activate dnn_bci
```


## EEG motor imagery datasets
Here two publicly available EEG BCI datasets are decoded: **5F** and **HaLT**. For the decoding analysis, the 19-EEG-channels signal is downsampled to 100Hz, and 150 trials are selected for each motor imagery condition. Each trial is epoched in the range [-250ms 1000ms] relative to onset.
The data along with the accompanying paper can be found at [(Kaya et al., 2018)][kaya].

#### 5F dataset
This is a motor imagery dataset of the 5 hand fingers movement: thumb, index finger, middle finger, ring finger, pinkie finger. The following files are used for the analyses:
1. _5F-SubjectA-160405-5St-SGLHand.mat_
2. _5F-SubjectB-160316-5St-SGLHand.mat_
3. _5F-SubjectC-160429-5St-SGLHand-HFREQ.mat_
4. _5F-SubjectE-160415-5St-SGLHand-HFREQ.mat_
5. _5F-SubjectF-160210-5St-SGLHand-HFREQ.mat_
6. _5F-SubjectG-160413-5St-SGLHand-HFREQ.mat_
7. _5F-SubjectI-160719-5St-SGLHand-HFREQ.mat_

To run the code, add the data files to the directory `/project_dir/datasets/5f/data/`.

[kaya]: https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698

#### HaLT dataset
This is a dataset consisting of 6 motor imagery conditions: left hand, right hand, middle finger, passive/neutral state, left leg, right leg, tongue. The following files are used for the analyses:
1. _HaLTSubjectA1602236StLRHandLegTongue.mat_
2. _HaLTSubjectB1602186StLRHandLegTongue.mat_
3. _HaLTSubjectC1602246StLRHandLegTongue.mat_
4. _HaLTSubjectE1602196StLRHandLegTongue.mat_
5. _HaLTSubjectF1602026StLRHandLegTongue.mat_
6. _HaLTSubjectG1603016StLRHandLegTongue.mat_
7. _HaLTSubjectI1606096StLRHandLegTongue.mat_
8. _HaLTSubjectJ1611216StLRHandLegTongue.mat_
9. _HaLTSubjectK1610276StLRHandLegTongue.mat_
10. _HaLTSubjectL1611166StLRHandLegTongue.mat_
11. _HaLTSubjectM1611086StLRHandLegTongue.mat_

To run the code, add the data files to the directory `/project_dir/datasets/halt/data/`.



## CNN model
The decoding analysis is performed using the shallow ConvNet architecture described in [Schirrmeister et al., 2018][schirrmeister].

[schirrmeister]: https://arxiv.org/abs/1703.05051v1



## Cropped trials
This is analogous to a data augmentation technique: instead of full trials, the CNN is fed with crops (across time) of the original trials. This procedure results in more training data, and has been shown to increase decoding accuracy. More information about cropped trials decoding in [Schirrmeister et al., 2018][schirrmeister], and a tutorial for the Python implementation of the method can be found on the [Braindecode][cropped_tutorial] website.

[cropped_tutorial]: https://braindecode.org/auto_examples/plot_bcic_iv_2a_moabb_cropped.html



## Inter-subject learning
Inter-subject learning is a zero-shot learning approach which aims at understanding how well a CNN trained on decoding the motor imagery trials of a set of subjects is capable of generalizing its decoding performance on a held-out subject. In other words, this is testing the possibility of pre-trained EEG BCI devices which readily work on novel subjects without the need of any training data from these subjects.



## Model training and results
The CNN models have been trained using the following parameters:

- **Learning rate:** 0.001
- **Weight decay:** 0.01
- **Batch size:** 128
- **Training epochs:** 500

Results are shown for the training epochs which yielded highest decoding accuracies:

![5F dataset results table](https://user-images.githubusercontent.com/50326481/120665912-7a39b580-c48c-11eb-91c7-e57ed7d47673.png)
![HaLT dataset results table](https://user-images.githubusercontent.com/50326481/120665901-786ff200-c48c-11eb-8f7d-0ebda8c4369c.png)