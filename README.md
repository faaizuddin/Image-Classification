# Image-Classification
A Computer Vision related task using Convolutional Neural Network (CNN) for Image Classification with CIFAR-10 dataset.

The same Image Classification program was run with different changes to know their effects on the accuracy. The results are shown in the graphs

For the first part, we see the affect of batch normalization in image classification on both training and testing accuracies. By looking at the graphs, it is a no brainer that batch normalization significantly helps the performance of the model.
![alt text](https://github.com/faaizuddin/Image-Classification/blob/master/TrAccBN.png)
![alt text](https://github.com/faaizuddin/Image-Classification/blob/master/TeAccBN.png)

Next up: effect of dropout. Dropout is a widely used regularization technique for neural networks to prevent overfitting. 
![alt text](https://github.com/faaizuddin/Image-Classification/blob/master/TrAccDrop.png)
![alt text](https://github.com/faaizuddin/Image-Classification/blob/master/TeAccBN.png)

The final part is to see the effects of various Optimizers. During the training process, we tweak and change the parameters (weights) of our model to try and minimize that loss function, and make our predictions as correct as possible. This is where we need optimizersand there are several of them.
GradientDescentOptimizer applies the  gradient descent algorithm. This happens to be the most basic one in the machine learning world.
RMSprop is a special version of Adagrad. Instead of letting all of the gradients accumulate for momentum, it only accumulates gradients in a fixed window. 
Adam is another way of using past gradients to calculate current gradients. Adam also utilizes the concept of momentum by adding fractions of previous gradients to the current one. This optimizer has become pretty widespread, and is practically accepted for use in training neural nets.
![alt text](https://github.com/faaizuddin/Image-Classification/blob/master/TrAccOpt.png)
![alt text](https://github.com/faaizuddin/Image-Classification/blob/master/TeAccOpt.png)
