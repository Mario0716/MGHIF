
## 
The codes are developed by lei zhu, Central South University
Contact: leizhu@csu.edu.cn / lzhu.cs@gmail.com

- This repository is the implementation of [MGHIF](https://dl.acm.org/doi/10.1145/3466641):
> Multi-Graph Heterogeneous Interaction Fusion for Social Recommendation
Chengyuan Zhang, Yang Wang, Lei Zhu, Jiayu Song, Hongzhi Yin

	


### Files in the folder
- `data/`: training and test data
- `results/`: evaluation results and the learned embeddings of the generator and the discriminator
- `src/`: source codes


### Requirements
The code has been tested running under Python 3.8.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.8.0
- tqdm == 4.23.4 (for displaying the progress bar)
- numpy == 1.14.3
- sklearn == 0.19.1
- torch == 1.7.1


### Input format

##### user-user graph data file sample
The user-user graph data is stored in data/\<Dataset>/trusts.txt.

```UserID UserID 1```  
```0	1 1```  
```3	2 1```  
```...```
##### user-item graph data file sample
The user-item graph data is stored in data/\<Dataset>/rating.txt, which is randomly sliced into test set and train set by a ratio of 3:7,2:8,1:9.

```UserID ItemID Rating```  
```0	13000 5```  
```1	20000 4```  
```...```




### Basic usage
Using GraphGAN with user-user graph data for pre-training.

```python pretrain.py```   

The backbone model first load the user embedding pretrained by GraphGAN, then use the divided user-item graph data to train. 
Note: the emb_dim in src/metapath/config.py should equal n_emb in src/GraphGAN/config.py.

```python main.py```

