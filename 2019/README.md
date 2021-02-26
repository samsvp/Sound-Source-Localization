# TAU Spatial Sound Events 2019 - Ambisonic and Microphone Array, Development Datasets

[Audio Research Group / Tampere University](http://arg.cs.tut.fi/)

Authors

- Sharath Adavanne (<sharath.adavanne@tuni.fi>, <https://scholar.google.com/citations?user=xCEvnG8AAAAJ&hl=en>)
- Archontis Politis (<archontis.politis@tuni.fi>, <https://scholar.google.fi/citations?user=DuCqB3sAAAAJ&hl=en>)
- Tuomas Virtanen (<tuomas.virtanen@tuni.fi>, <http://www.cs.tut.fi/~tuomasv/>)

Recording and annotation

- Eemi Fagerlund
- Aino Koskimies
- Aapo Hakala

This package consists of two development datasets, **TAU Spatial Sound Events 2019 - Ambisonic** and **TAU Spatial Sound Events 2019 - Microphone Array**. These datasets contain recordings from an identical scene, with **TAU Spatial Sound Events 2019 - Ambisonic** providing four-channel First-Order Ambisonic (FOA) recordings while  **TAU Spatial Sound Events 2019 - Microphone Array** provides four-channel directional microphone recordings from a tetrahedral array configuration. Both formats are extracted from the same microphone array. The recordings in the two datasets consists of stationary point sources from multiple sound classes each associated with a temporal onset and offset time, and DOA coordinate represented using azimuth and elevation angle. These development datasets are part of the [DCASE 2019 Sound Event Localization and Detection Task](https://github.com/sharathadavanne/seld-dcase2019).


Both the development set consists of 400, one minute long recordings sampled at 48000 Hz, and divided into four cross-validation splits of 100 recordings each. These recordings were synthesized using spatial room impulse response (IRs) collected from five indoor locations, at 504 unique combinations of azimuth-elevation-distance. Furthermore, in order to synthesize the recordings the collected IRs were convolved with [isolated sound events dataset from DCASE 2016 task 2](http://www.cs.tut.fi/sgn/arg/dcase2016/task-sound-event-detection-in-synthetic-audio#audio-dataset). Finally, to create a realistic sound scene recording, natural ambient noise collected in the IR recording locations was added to the synthesized recordings such that the average SNR of the sound events was 30 dB.

The IRs were collected in Finland by Tampere University between 12/2017 - 06/2018. The data collection received funding from the European Research Council, grant agreement 637422 EVERYSOUND.

[![ERC](https://erc.europa.eu/sites/default/files/content/erc_banner-horizontal.jpg "ERC")](https://erc.europa.eu/)

## Recording Procedure

The real-life IR recordings were collected using an [Eigenmike](https://mhacoustics.com/products) spherical microphone array. A [Genelec G Two loudspeaker](https://www.genelec.com/home-speakers/g-series-active-speakers) was used to playback a maximum length sequences (MLS) around the Eigenmike. The MLS playback level was ensured to be 30 dB greater than the ambient sound level during the recording. The IRs were obtained in the STFT domain using a least-squares regression between the known measurement signal (MLS) and far-field recording independently at each frequency. These IRs were collected in the following directions:

- 36 IRs at every 10&deg; azimuth angle, for 9 elevations from -40&deg; to 40&deg; at 10&deg; increments, at 1 m distance from the Eigenmike, resulting in 324 discrete DOAs.
- 36 IRs at every 10&deg; azimuth angle, for 5 elevations from -20&deg; to 20&deg; at 10&deg; increments, at 2 m distance from the Eigenmike, resulting in 180 discrete DOAs.

The IRs were recorded at five different indoor locations inside the Tampere University campus at Hervanta, Finland. Additionally, we also collected 30 minutes of ambient noise recordings from these five locations with the IR recording setup unchanged. The description of the indoor locations are as following:

- Language Center - Large common area with multiple seating tables and carpet flooring. People chatting and working.
- Reaktori Building - Large cafeteria with multiple seating tables and carpet flooring. People chatting and having food.
- Festia Building - High ceiling corridor with hard flooring. People walking around and chatting.
- Tietotalo Building - Corridor with classrooms around and hard flooring. People walking around and chatting.
- Sahkotalo Building - Large corridor with multiple sofas and tables, hard and carpet flooring at different parts. People walking around and chatting.


## Recording format and dataset specifications

The [isolated sound events dataset from DCASE 2016 task 2](http://www.cs.tut.fi/sgn/arg/dcase2016/task-sound-event-detection-in-synthetic-audio#audio-dataset) consists of 11 classes, each with 20 examples. The sound classes are clearthroat, cough, doorslam, drawer, keyboard, keys drop, knock, laughter, pageturn, phone and speech. The 220 examples (11 classes * 20 examples) are randomly split into five sets with an equal number of examples for each class, the first four sets are used for synthesizing the four splits of development dataset, while the remaining one set is used for evaluation dataset that will be released as a separate package. For each split of the dataset, we synthesize 100 recordings. Each of these recordings is generated by randomly choosing sound event examples from the corresponding set and assigning a start time, and one of the collected IRs randomly. Finally, by convolving each of these assigned sound examples with their respective IRs, we spatially position them at a given distance, azimuth and elevation angles from the Eigenmike. We make sure to use IRs from a single location for all sound events in a recording. Further, half of the recordings in each split are synthesized with up to two temporally overlapping sound events while the others are synthesized with no overlapping sound events. Finally, the ambient noise collected at the respective IR location was added to the synthesized recording such that the average SNR of the sound events is 30 dB.

Since the number of channels in the IRs is equal to the number of microphones in Eigenmike (32), in order to create the **TAU Spatial Sound Events 2019 - Microphone Array** dataset we use the channels 6, 10, 26, and 22  that corresponds to microphone positions (45&deg;, 35&deg;, 42cm), (-45&deg;, -35&deg;, 42cm), (135&deg;, -35&deg;, 42cm) and (-135&deg;, 35&deg;, 42cm). The spherical coordinate system in use is right-handed with the front at (0&deg;, 0&deg;), left at (90&deg;, 0&deg;) and top at (0&deg;, 90&deg;). Further, the **TAU Spatial Sound Events 2019 - Ambisonic** dataset is obtained by converting the 32 channel microphone signals to FOA, by means of encoding filters based on anechoic measurements of the Eigenmike array response.

In summary, there are 100 recordings in each of the four splits of the development dataset. These 100 recordings are comprised of 10 recordings that have either up to two, or no temporally overlapping sound events, synthesized using the IRs from the five locations (10 * 2 * 5 = 100). Each of the development dataset splits consists of IRs from all the five locations, the dataset only guarantees a balanced distribution of sound events in each of the 36 azimuths and 9 elevation angles within the splits but does not guarantee the use of IRs collected at a single location to be entirely present in a single split. For example, some of the IRs of  Reaktori building might not be in the first split but might occur in any of the other splits.

More details on the IR recordings collections and synthesis of the dataset can be read [here](https://arxiv.org/pdf/1807.00129.pdf).


## Naming Convention
The recordings in the development dataset follow the naming convention:

    split[number]_ir[location number]_ov[number of overlapping sound events]_[recording number per split].wav

The information of the location whose impulse response has been used to synthesize the recording or the number of overlapping sound events in the recording is only provided for the participant to understand the performance of their method with respect to different conditions.


## Reference labels
As labels, for each recording in the development dataset, we provide a CSV format file with the same name as the recording and `.csv` extension. This file enlists the sound events, their respective temporal onset-offset times, azimuth and elevation angles. These CSV files are common for both the **TAU Spatial Sound Events 2019 - Ambisonic** and **TAU Spatial Sound Events 2019 - Microphone Array** development datasets.


## Task setup
The development dataset consists of a pre-defined four cross-validation split as shown in the table below. These splits consist of audio recordings and the corresponding metadata describing the sound events and their respective locations within each recording. Participants are required to report the performance of their method on the testing splits of the four folds. In order to allow a fair comparison of methods on the development dataset modification of the defined splits is not allowed. 

|   Folds   | Training splits | Validation split | Testing split |
|-----------|:---------------:|:----------------:|:-------------:|
|**Fold 1** |       3, 4      |        2         |       1       |
|**Fold 2** |       4, 1      |        3         |       2       |
|**Fold 3** |       1, 2      |        4         |       3       |
|**Fold 4** |       2, 3      |        1         |       4       |


## File structure

```
dataset root
│   README.md				this file, markdown-format
│   README.html				this file, html-format
│   LICENSE.md				also copied in the end of this file
│
└───foa_dev				Ambisonic format, 400 audio segments, 48kHz, four channels
│   │  	split1_ir0_ov1_1.wav 		file naming convention: split[number]_ir[location number]_ov[number of overlapping sound events]_[recording number per split].wav
│   │	split1_ir0_ov1_2.wav
│   │	...
│   │	split1_ir1_ov1_21.wav
│   │	split1_ir1_ov1_22.wav
│   │	...
│   │	split4_ir4_ov1_81.wav
│   │	split4_ir4_ov1_81.wav
│   │   ...
│
└───mic_dev				Microphone array format, 400 audio segments, 48kHz, four channels
│   │  	split1_ir0_ov1_1.wav 		file naming convention: split[number]_ir[location number]_ov[number of overlapping sound events]_[recording number per split].wav
│   │	split1_ir0_ov1_2.wav
│   │	...
│   │	split1_ir1_ov1_21.wav
│   │	split1_ir1_ov1_22.wav
│   │	...
│   │	split4_ir4_ov1_81.wav
│   │	split4_ir4_ov1_81.wav
│   │   ...
│
└───metadata_dev			`csv` format, 400 files, each lists the sound events, their respective temporal onset-offset times, and DOA in azimuth and elevation angles.
    │  	split1_ir0_ov1_1.csv 		file naming convention: split[number]_ir[location number]_ov[number of overlapping sound events]_[recording number per split].csv
    │	split1_ir0_ov1_2.csv
    │	...
    │	split1_ir1_ov1_21.csv
    │	split1_ir1_ov1_22.csv
    │	...
    │	split4_ir4_ov1_81.csv
    │	split4_ir4_ov1_81.csv
    │   ...


```
## Download

The three files,  `foa_dev.z01`, `foa_dev.z02` and `foa_dev.zip`, correspond to audio data of **TAU Spatial Sound Events 2019 - Ambisonic** development dataset.
The two files, `mic_dev.z01` and, `mic_dev.zip`, correspond to audio data of **TAU Spatial Sound Events 2019 - Microphone Array** development dataset.
The `metadata_dev.zip` is the common metadata for both **TAU Spatial Sound Events 2019 - Ambisonic** and **TAU Spatial Sound Events 2019 - Microphone Array** development datasets.

Download the zip files corresponding to the dataset of interest and use your favorite compression tool to unzip these split zip files.

For example, on linux, after downloading the zip files, we first merge all the split zip files on our local machine. This merged file is then unzipped to obtain the files. 
For the **TAU Spatial Sound Events 2019 - Ambisonic** development dataset we can use the following commands to get the files

```
 zip -s 0 foa_dev.zip --out unsplit_foa_dev.zip
 unzip unsplit_foa_dev.zip
```
or the following commands for **TAU Spatial Sound Events 2019 - Microphone Array** development dataset 
```
 zip -s 0 mic_dev.zip --out unsplit_mic_dev.zip
 unzip unsplit_mic_dev.zip
```


## License
Copyright (c) 2019 Tampere University and its licensors
All rights reserved.
Permission is hereby granted, without written agreement and without license or royalty
fees, to use and copy the TAU Spatial Sound Events 2019 - Ambisonic and Microphone Array,
development dataset (“Work”) described in this document and composed of audio and metadata. 
This grant is only for experimental and non-commercial purposes,
provided that the copyright notice in its entirety appear in all copies of this Work,
and the original source of this Work, (Audio Research Group at Tampere University),
is acknowledged in any publication that reports research using this Work.

Any commercial use of the Work or any part thereof is strictly prohibited.
Commercial use include, but is not limited to:
- selling or reproducing the Work
- selling or distributing the results or content achieved by use of the Work
- providing services by using the Work.

IN NO EVENT SHALL TAMPERE UNIVERSITY OR ITS LICENSORS BE LIABLE TO ANY PARTY
FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE
OF THIS WORK AND ITS DOCUMENTATION, EVEN IF TAMPERE UNIVERSITY OR ITS
LICENSORS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

TAMPERE UNIVERSITY AND ALL ITS LICENSORS SPECIFICALLY DISCLAIMS ANY
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE. THE WORK PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND
THE TAMPERE UNIVERSITY HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
