# MCCL
[***2025] Code for our paper MCCL : ***

## Prerequisites
- Python3
- numpy == 1.24.4
- PyTorch == 1.13.1 (with CUDA and CuDNN (cu116))
- torchvision==0.14.1 (cu116)
- scikit-learn == 1.3.2
- xgboost == 2.0.3

Please create and activate the following conda environment. To reproduce our results, please kindly create and use this environment.

```python
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate mccl
```

## Train MCCL model
The program can be run with the default parameters using the following:

```python
#Train for DAIC-WOZ
cd mccl
python main2017.py

#Train for E-DAIC
cd mccl
python main2019.py

#Train for CMDC
cd mccl
python cmdc.py

```

## Test MCCL model
The code was tested on an RTX 3090.
We provide the checkpoint files on DAIC-WOZ dataset [here](https://drive.google.com/drive/folders/1QKpPl7ng004bH6iqPpUNJomcPHFVWyLF?usp=sharing)

Please follow the below steps to test the MCCL model on diverse datasets.
1. Download checkpoint files and put them in `mccl/checkpoint/{dataset_name}` folder.
2. run `python main.py --dataset={dataset_name} --inference=1` to test the model on diverse datasets, the dateset_name can be set to 'DAIC', 'EDAIC', 'CMDC'.

## Citation
Please cite our work if you find it useful.
```bibtex
-----------------
```

## Acknowledgement
The data processing of different datasets is based on the code [DepressionEstimation(DAIC)](https://github.com/PingCheng-Wei/DepressionEstimation.git), [AVEC2019(EDAIC)](https://github.com/AudioVisualEmotionChallenge/AVEC2019.git), [CMDC](https://github.com/CMDC-corpus/CMDC-Baseline.git). 

## Data links
DAIC-WOZ and E-DAIC can be applied from [here](https://dcapswoz.ict.usc.edu/)

CMDC can be applied from [here](https://ieee-dataport.org/open-access/chinese-multimodal-depression-corpus).

The files should be unzipped, and the features should be extracted from the unzipped folder according to the code in 'mccl/data_processing'.

## Contact
For questions regarding the code, please contact wangxinkecn@163.com.
