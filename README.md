# IJCAI-PRICAI 2020 3D AI Challenge: Image-based 3D Shape Retrieval 

## Data Preparation 

*Data Preparation

* Please make sure that the training and testing data have been downloaded and put into "dataset/train_data" and "dataset/test_data", respectively.

* For representing a 3D shape, we utilized 5 typical views of a 3D shape, which are rendered based on the official Toolbox. The following images represent the rendered views of 0000001.obj. 

  <center class='half'>
      <img src="dataset\examples\0000005.png" alt="0000005" style="zoom:10%;" />
      <img src="dataset\examples\0000005.png" alt="0000006" style="zoom:10%;" />
      <img src="dataset\examples\0000007.png" alt="0000007" style="zoom:10%;" />
      <img src="dataset\examples\0000008.png" alt="0000008" style="zoom:10%;" />
      <img src="dataset\examples\0000009.png" alt="0000009" style="zoom:10%;" />
  <center>

 
### Training

Unlike the baseline method, which divides the training process into two stages, we directly train our proposed network for one stage, considering modality-specific feature learning and cross-modality alignment at the same time. 

```
bash train_workshop_baseline_tuning.sh
```
**Note:** 

modify ``CUDA_VISIBLE_DEVICES=0,1``  to assign the GPU IDs;



**Our best-trained model:**

**Our best-trained model is in the folder "checkpoints/workshop_baseline_notexture_tuning_v1"**

### Test

**Feature Extraction and Matching**

```
# extract 2D feature, generate folder: extract_workshop_baseline_notexture_2d_v1
bash extract_workshop_baseline_2d.sh

# extract 3D feature, generate folder: extract_workshop_baseline_notexture_3d_v1
bash extract_workshop_baseline_3d.sh

# matching 2D-3D, generate retrieval_results.txt
# default using euclidean distance
python eval_workshop_baseline.py

# prepare submit data for evaluation
zip retrieval_results.zip retrieval_results.txt
```

## Online Result

Our online result for baseline is shown as below. 
```
"score": 51.11, "top1_acc": 0.53, "mean_f_score": 48.67
```


