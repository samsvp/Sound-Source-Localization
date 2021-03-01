# Fast Angle of Arrival

The fast angle of arrival is an implementation of the frequency domain beamforming and time domain beamforming algorithms to localizate one sound source in an environment.

# Table of Contents
1. [SELD](#SELD)
2. [SELD 2019 Dataset](#SELD-2019-dataset)
3. [Sound Localization Task](#Sound-Localization-Task)

## SELD
[The Sound Event Localization and Detection (SEDL)](http://dcase.community/challenge2019/task-sound-event-localization-and-detection) is challenge where competitors need to recognize the temporal onset and offset of sound events when active, classifying the sound events into known set of classes, and further localizing the events in space when active.

We are mainly interested in the sound localization task and will ignore the sound classification part of the challenge.

## SELD 2019 dataset
The SELD 2019 dataset can be found [here](https://zenodo.org/record/2599196#.YDk1dmhKjIU). We will be using the data inside `mic_dev.z01` and `mic_dev.zip`, which is the data collected by an array of four microphones. Download both files and extract it inside the 2019 folder.

To preprocess the dataset we have the `preprocess.py` script, which loops through each `.WAV` file and gets only the time stamps which contain some sort of sound and saves the sound arrays into `json` files for a easier usage.

## Sound Localization Task
For the localization task we will be using the findings contained in the following article: [Sound Event Localization and Tracking](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-and-tracking). We will explore direction of arrival (DOA) estimation as a classification and regression task with the end goal being the usage of a more lightweight algorithm than convolutional neural network to use inside autonomous systems such as autonomous underwater vehicles (AUVs) and unmanned aerial vehicles(UAVs). The input of the algorithms will be the spectogram of the sound waves as detailed [here](https://github.com/sharathadavanne/seld-dcase2019#more-about-seldnet).
