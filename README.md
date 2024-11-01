# CAMP
The official code for the paper "CAMP: A Cross-View Geo-Localization Method using Contrastive Attributes Mining and Position-aware Partitioning".

The current version of the repository can cover the experiments reported in the paper, for researchers in time efficiency. And we will also update this repository for better understanding and clarity.

## 1. For University-1652 dataset.

Train: run *train_university.py*, with --only_test = False.

Test: run *train_university.py*, with --only_test = True, and choose the model in --ckpt_path.



## 2. For SUES-200 dataset.

You need to split the origin dataset into the appropriate format using the script "CAMP-->sample4geo-->dataset-->SUES-200-->split_datasets.py".

The processed format should be:

```
├─ SUES-200
  ├── Training
    ├── 150/
    ├── 200/
    ├── 250/
    └── 300/
  ├── Testing
    ├── 150/
    ├── 200/ 
    ├── 250/	
    └── 300/
```

The train and test operation is similar to the University-1652 dataset but with the script train_sues200.py


## 3. Models

We provide the trained model for University-1652 in the link below:
https://drive.google.com/file/d/1qHjXr3VVQuJZ5kE5u7YrUB8id90Nv2GJ/view?usp=sharing

and the trained models for SUES-200:

for 150m: https://drive.google.com/file/d/14ybgPvezIP9Yv9QOGzYbS-YpRj738p8u/view?usp=sharing

for 200m: https://drive.google.com/file/d/1D3IZ209quCbyLq5Gib-ZUjQN2WBc8smA/view?usp=sharing

for 250m: https://drive.google.com/file/d/1Tvbz0D24uVHD2VK8KBGgBnfA3QCm3lXr/view?usp=sharing 

for 300m: https://drive.google.com/file/d/1pjY1ubfyvFbITB1c6n-I7cdp7QHJv0WD/view?usp=sharing

We will update this repository for better clarity ASAP, current version is for quick research for researchers interested in the cross-view geo-localization task.

## 4. Acknowledgement
This repository is built using the Sample4Geo[https://github.com/Skyy93/Sample4Geo] and MCCG[https://github.com/mode-str/crossview] repositories.

