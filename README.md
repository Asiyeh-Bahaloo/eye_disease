# AI_Medic

## Abstraction
<p align="justify"> As retinal pathologies are becoming more and more common globally, rapid and accuract detecting of eye diseases plays a crucial role in preventing from blindness. Therefore deep learning approaches will sharply increase probability of curing such diseases.


## Introduction
According to [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment) at least 2.2 billion people have a near or distance vision impairment. In at least 1 billion – or almost half – of these cases, vision impairment could have been prevented or has yet to be addressed.

This 1 billion people includes those with moderate or severe distance vision impairment or blindness due to unaddressed refractive error (88.4 million), cataract (94 million), glaucoma (7.7 million), corneal opacities (4.2 million), diabetic retinopathy (3.9 million), and trachoma (2 million), as well as near vision impairment caused by unaddressed presbyopia (826 million). Population growth and ageing are expected to increase the risk that more people acquire vision impairment. 

![Image alt text](/img/prev.JPG)


In the past few years , deep learning has helped lots of people to live better around the world. In this case we will use deep learning to detect different eye pathologies. Some critical articles in this area were studied.
Some articles used preprocessing methods to improve the image. Then they trained several models to get features and finaly they predicted the potential disease. Some other articles developed models to detect some lesions such as microaneurysms, hemorrhages, exudates, and cotton-wool spots.

## Datasets
We have used two datasets to train our models:
1. ODIR_2019 dataset [(downlaod)](https://odir2019.grand-challenge.org/dataset/):  
   This dataset is real-life set of patient information collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China. In these institutions, fundus images are captured by various cameras in the market, such as Canon, Zeiss and Kowa, resulting into varied image resolutions. 

2. Cataract dataset [(downlaod)](https://www.kaggle.com/jr2ngb/cataractdataset): 
     Cataract and normal eye image dataset for cataract detection.

After exploring datasets, we used some preprocessing techniques to improve images: 
1. Resize images : We resized images to 224*224 pixels
2. Remove padding : We removed the padding around the fundas image and cropped uninformative area to detect lesions better.
3. Ben_Graham Method : Ben Graham (Kaggle competition's winner) share insightful way to improve lighting condition. Here, we applied his idea, and we could see many important details in the eyes much better. 
## Methodology
Firstly we explored datasets. 
Then we preprocessed datasets with methods mentioned in Dataset section.
Next we trained five different deep learning models to classify images into 8 groups to extract relevant image features and automatic detection of eye diseases in fundas photographs. Our models: 
1. VGG16
2. VGG19
3. Resnet_V2
4. Inception_V3
5. Xception

Finally, different results from the experiments were generated and compared using MLflow.

## Results
In this section we will show and compare the results of our models.
* VGG16
   We runned VGG16 model for 50 epochs with a batch size of four. Training data had 5896 images and validation data included 655 validation images. Also there was 1000 images for testing VGG16 model.
   Here are the results on test data: 
   Training:

| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.8939   | 0.9061 | 0.2444 | 0.6380    | 0.4716 |
 
   Validation:

| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.8326   | 0.7990 | 0.3813 | 0.3757    | 0.2374 |


* VGG19
   We runned VGG19 model for 50 epochs with a batch size of four. Training data had 5896 images and validation data included 655 validation images. Also there was 1000 images for testing VGG19 model.
   Here are the results on test data: 
   Training:

| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.8910   | 0.9046 | 0.2492 | 0.6201    | 0.4692 |


   Validation:

| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.8498   | 0.7942 | 0.4214 | 0.4527    | 0.1833 |


* Resnet_V2

   We runned Resnet_V2 model for 100 epochs with a batch size of four. Training data had 5896 images and validation data included 655 validation images. Also there was 1000 images for testing Resnet_V2 model.
   Here are the results on test data:  

   Training: 

| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.9511   | 0.9771 | 0.1249 | 0.8721    | 0.7449 |

   Validation:

| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.8403   | 0.7779 | 0.5965 | 0.3778    | 0.3559 |


* Inception_V3

   We runned Inception_V3 model for 100 epochs with a batch size of four. Training data had 5896 images and validation data included 655 validation images. Also there was 1000 images for testing Inception_V3 model.
   Here are the results on test data:  

   Training: 

| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.8660   | 0.7993 | 0.3166 | 0.4628    | 0.0337 |

   Validation:

| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.8553   | 0.5603 | 16.198 | 0         | 0      |


* Xception
   We runned Xception model for 100 epochs with a batch size of four. Training data had 5896 images and validation data included 655 validation images. Also there was 1000 images for testing Xception model.
   Here are the results on test data:  

   Training:
    
| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.9320   | 0.9566 | 0.1691 | 0.8190    | 0.6289 |

   Validation:

| accuracy | auc    | loss   | precision | recall |
| -------- | ------ | ------ | --------- | ------ |
| 0.8660   | 0.8435 | 0.3380 | 0.5469    | 0.3535 |


* All in One
  
  **Training**:


  **auc plot:** 
  ![Image alt text](/img/t_auc.JPG)


  **precision plot:**  
  ![Image alt text](/img/t_precision.JPG)


  **recall plot:** 
  ![Image alt text](/img/t_recall.JPG)


   **Validation**:

     **auc plot:** 
  ![Image alt text](/img/v_auc.JPG)


  **precision plot:**  
  ![Image alt text](/img/v_precision.JPG)


  **recall plot:** 
  ![Image alt text](/img/v_recall.JPG)
## Conclusion

* This project studied five deep learning models for the multiple classification of diseases.
* We faced several challenges due to the initial data imbalance.
