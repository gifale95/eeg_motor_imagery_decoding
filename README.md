# Decoding Motor Imagery states from EEG data using Convolutional Neural Networks
This repository provides Python code for the decoding of different motor imagery conditions from raw EEG data, using CNNs. Install the following packages to run the analyses:
1. Braindecode from https://github.com/braindecode/braindecode.
2. MNE-Python from https://github.com/mne-tools/mne-python.



## EEG motor imagery datasets
#### BCI Competition IV dataset 2a
Motor imagery dataset (Brunner et al., 2008) of 9 subjects with 4 motor imagery coditions: left hand, right hand, both feet, tongue. Each condition has 144 four-second trials, recorded using 22 EEG channels. Data and the accompanying paper can be found at http://bbci.de/competition/iv/#dataset2a.

#### 5F dataset
9 subjects motor imagery dataset (Kaya et al., 2018) of the 5 hand fingers movement: thumb, index finger, middle finger, ring finger, pinkie finger. Each condition has 150 one-second trials, recorded using 19 EEG channels. Data and the accompanying paper can be found [here][kaya].
The following data files were used for the analysis:
1. _5F-SubjectA-160408-5St-SGLHand-HFREQ.mat_
2. _5F-SubjectB-160309-5St-SGLHand-HFREQ.mat_
3. _5F-SubjectC-160429-5St-SGLHand-HFREQ.mat_
4. _5F-SubjectE-160321-5St-SGLHand-HFREQ.mat_
5. _5F-SubjectF-160210-5St-SGLHand-HFREQ.mat_
6. _5F-SubjectG-160413-5St-SGLHand-HFREQ.mat_
7. _5F-SubjectH-160804-5St-SGLHand-HFREQ.mat_
8. _5F-SubjectI-160719-5St-SGLHand-HFREQ.mat_

[kaya]: https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698

#### HaLT dataset
12 subjects dataset of 6 motor imagery conditions: left hand, right hand, middle finger, passive/neutral state, left leg, right leg, tongue. Each condition has 150 one-second trials, recorded using 19 EEG channels. Data and the accompanying paper can be found at the same link as the 5F dataset (see above).
The following data files were used for the analysis:
1. _HaLTSubjectA1602236StLRHandLegTongue.mat_
2. _HaLTSubjectB1602186StLRHandLegTongue.mat_
3. _HaLTSubjectC1602246StLRHandLegTongue.mat_
4. _HaLTSubjectE1602196StLRHandLegTongue.mat_
5. _HaLTSubjectF1602026StLRHandLegTongue.mat_
6. _HaLTSubjectG1603016StLRHandLegTongue.mat_
7. _HaLTSubjectH1607206StLRHandLegTongue.mat_
8. _HaLTSubjectI1606286StLRHandLegTongue.mat_
9. _HaLTSubjectJ1611216StLRHandLegTongue.mat_
10. _HaLTSubjectK1610276StLRHandLegTongue.mat_
11. _HaLTSubjectL1611166StLRHandLegTongue.mat_
12. _HaLTSubjectM1611086StLRHandLegTongue.mat_



## CNN models
The decoding analysis is performed using shallow and deep CNN architectures, as described in [Schirrmeister et al., 2018][schirrmeister].

[schirrmeister]: https://arxiv.org/abs/1703.05051v1



## Inter-subject learning



## Cropped trials










