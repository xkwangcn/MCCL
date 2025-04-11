# MCCL
[TCSVT 2025] Code for our paper MCCL: Facial Depression Estimation via Multi-Cue Contrastive Learning

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
python main.py --dataset='DAIC' --output_name='daic'
```

## Test MCCL model
The code was tested on an RTX 3090.

Please follow the below steps to test MCCL model on DAIC-WOZ dataset.
1. We have put checkpoint files in `mccl/checkpoint/DAIC` folder.
2. run `python main.py --dataset='DAIC' --inference=1` to test the model on DAIC-WOZ dataset.


## Citation
Please cite our work if you find it useful.
```bibtex
@ARTICLE{10852375,
  author={Wang, Xinke and Xu, Jingyuan and Sun, Xiao and Li, Mingzheng and Hu, Bin and Qian, Wei and Guo, Dan and Wang, Meng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Facial Depression Estimation via Multi-Cue Contrastive Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Depression;Estimation;Contrastive learning;Visualization;Correlation;Facial features;Circuits and systems;Three-dimensional displays;Feature extraction;Interviews;Facial depression estimation;multi-cue;contrastive learning},
  doi={10.1109/TCSVT.2025.3533543}}
```

## Acknowledgement
The data processing of DAIC-WOZ dataset is based on the code [DepressionEstimation(DAIC)](https://github.com/PingCheng-Wei/DepressionEstimation.git). 

## Data links
The Dataset can be applied from: [here](https://dcapswoz.ict.usc.edu/)

The files should be unzipped, and the features should be extracted from the unzipped folder according to the code in 'mccl/data_processing'.
