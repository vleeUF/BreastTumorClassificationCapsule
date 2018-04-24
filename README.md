# Breast Tumor Classification Using Capsule Network

## Dataset
The breast tumor dataset consists of 9,109 histopathological images collected from 82 patients using four different magnifying factors (40x, 100x, 200x, 400x). Each image is composed of 700x460 pixels in 3 channel RGB in a PNG format. The dataset consists of eight breast tumor types with four benign (adenosis, fibroadenoma, phyllodes tumor, and tubular adenoma) and four malignant (ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma). There are 2,480 benign and 5,429 malignant samples. 

The dataset was taken from: https://www.ncbi.nlm.nih.gov/pubmed/26540668

## Capsule Networks
Convolutional Neural Networks (CNN) have been a huge success with deep learning, specifically computer vision, as computing power has grown exponentially over the recent years. However, there are still challenges that are met when utilizing these models. One such issue is that information is thrown away about precise positions of entities in the images being processed during max-pooling. This makes CNNs vulnerable to affine transformations like skew or rotation. The way CNNs are built tend to force these models to “memorize” the pixels in the image to classify objects rather than break down the components of the image and classify entities from its components and their respective positions sort of the way human vision works.

Different engineering tricks have been applied to get around this but the fundamental flaw of the architecture has not been directly addressed until recently with the routing algorithm between capsules (Hinton et al. [2017]). Each layer of the CNN is composed of capsules rather than neurons. The fundamental difference is that capsules act as groups of neurons thus use matrix multiplication of input vectors rather than scalar weighting of scalar input, and are then weighted and summed to be applied to a vector-to-vector nonlinearity activation function. 

The routing algorithm itself allows lower level capsules to send its input to the higher-level capsule that is most similar. This is achieved by calculating the dot product of the two outputs of the capsules as a measure of similarity. A negative dot product would indicate that the capsules are not similar and decrease the routing coefficient, a temporary value that is iteratively updated to determine where to “send” the output of the lower level capsule to the higher-level capsule. A positive dot product would have the opposite effect of increasing the routing coefficient matching the capsules from the two levels. Iterating too many times may result in overfitting with three being the recommended number of iterations.

## Results
Testing Accuracy is currently only at 76% so improvement is definately still need. However, the model is producing respectable results considering the images are being resized 10% of its original size. The model is currently being modified for further performance improvement.

