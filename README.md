# depth-2d-keras
Deep Learning model on Jupyter notebook to get depth data from 2d images directly. Implemented on tensorflow keras.

## Dataset
This dataset is collected from the Record 3D app available on the app store, using USB streaming. The dataset provides us with images and depth maps separately which are both respectively X and Y for our dataset. The dataset consists of 1059 samples of RGB and depth map. The depth map is produced from LiDAR Scanner from the iPhone. Each image and depth map is of resolution 720 × 960, with 3 RGB channels in 2D image and 1 depth channel in depth map image.

## Approach

We initially started with a single neural network which takes input as RGB image, I and outputs the depth map D. The problem can be translated as:

D = f(I),    where D ∈ (R, R, R) and I ∈ (R, R, (R, R, R))
where input I is an RGB image with three channels, red, green and blue, values lying between 0 to 255. The depth map has one channel,  with values lying between 0 to 65535. We choose mean squared error as our loss function.


The neural network consists of 8 conv2d layers, each with 3 x 3 kernel, 0 padding, relu as activation function, with batch normalization at each step. The number of filters are 50, 80, 120, 120, 80, 50, and the final layer with 1 filter. There is one conv2d transpose layer which does the reverse operation of conv2d. The main intuition behind using this layer is that the some lower level feature, i.e., super pixel might have sufficient information to infer about depth, therefore we can decompose the information of that superpixel to other surrounding pixels.

To improve upon the results, we further deployed the model with two neural networks:

* Gradient network - The results of the above model indicated that the edge detection could be improved, which can be fed into the original image to get a better result. Moreover, edges, vanishing points give a strong cue about the Therefore, in this model,  we train it beforehand to learn about the cues in the depth map. The input of this model is the RGB image. The ground truth of this network is created by running a sobel detection algorithm on the depth map, the output of which is a grayscale image with depth information. The neural network has similar model parameters and hyperparameters to the model above with MSE as the Loss function.
* Finetune network - The results of the gradient network model and the original RGB image are then fed into an original neural network, to learn the depth information. The result of the gradient network is concatenated with the results of the first layer of the network at the second layer and further model layers remain the same as above.

## Inference

The challenge of recovering depth information from a single 2D image is not a trivial task and requires a contextual understanding of the image to infer about the depth. There are a lot of challenges like scale-invariance issues, noise issues etc. Though the model runs well on a local dataset, it provides a good performance on other dataset. This is an indication of over-fitting. 

Metrics are an important factor to evaluate the model. Initially, I started with accuracy as the key metrics, but accuracy was too low. Therefore, I moved on to keep mean squared loss as a metric to evaluate the model. The mean squared loss provides the evaluation of performance of the model on a global scale, rather than pixel scale. One reason why the model is not accurate is that it's actually hard to infer about the depth information just by looking at the image. Therefore, the complexity of the output depth map is little related to the rgb image and makes it difficult to find patterns and give good accuracy.

Since computer vision is hard, computer vision tasks should be solved using first principles as a toolbox. The idea about gradient networks to identify the cues can only be realized when we think in terms of basics, in terms of mathematics, and specifying the model for what to learn. The neural network loses the global information at each new layer.

The choice of activation function has a huge impact on the model. Initially, I started with keeping tanh as the model however, the mean squared errors were 10x the results above. I experimented with multiple activation functions, to realize that relu is the best choice for us.
The choice of learning rate is important. In the model, I experimented with learning rates from 1e-4, 0.01 0.1. While, 1e-4 was too slow in converging to the minima, 0.1 was so big that I suffered from unexpected divergent behavior. Therefore, 0.01 is the optimal choice for our model.

## Thank you :)


