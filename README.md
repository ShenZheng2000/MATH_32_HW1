# MATH_32_HW1

# Overview

In this assignment, you will implement the [LeNet](https://en.wikipedia.org/wiki/LeNet#:~:text=In%20general%2C%20LeNet%20refers%20to,in%20large%2Dscale%20image%20processing.) convolutional neural network (convnets) using [PyTorch](https://pytorch.org/) for [MNIST](http://yann.lecun.com/exdb/mnist/) digit recognition. You will also design relevant loss functions, and configure suitable optimizers. Besides, you will gain experience in hyperparameter tuning, specifically focusing on adjusting the batch size and learning rate. The ultimate goal is to give you practical exposure to building and optimizing a deep learning model, providing a solid foundation for future, more complex tasks.

# Installation

* Install Visual Studio Code (VS Code) from [here](https://code.visualstudio.com/)
* Install Python from VS Code extensions. 
* Install the following packages (if you already has Anaconda, you can skip this step)
  ```
  pip install scikit-learn
  pip install matplotlib
  pip install numpy
  ```
* Install PyTorch-CPU. 
  ```
  pip install torch torchvision torchaudio
  ```

# Code Completion

* Open `main_q1.py`, complete 4 TODO's. 
* For the first TODO, you should complete the feature extraction layers for LeNet.
* For the second TODO, you should complete the classification layers for LeNet.
* For the third TODO, you have to define the cross-entropy loss.
* For the last TODO, you have to define the Adam optimizer with the specified learning rate.

# Training and Testing

* Run in terminal:
  ```
  python main_q1.py --model lenet
  ```
* You should find a folder named `lenet`, and a subfolder named `bs64_lr0.001` under `lenet`.
* Inside that subfolder, there will be three images: `confusion.png`, `loss.png`, and `t-SNE.png`

# Hyperparameter Tuning

* Adjust the batch size and the learning rate to see how the model performance changes.
* Note that you should keep the three images for each experiment.
* For the experiments with different batch sizea, run the following command in the terminal:
  ```
  python main_q1.py --model lenet --batch_size 16
  python main_q1.py --model lenet --batch_size 32
  python main_q1.py --model lenet --batch_size 64
  ```
  
* For the experiments with different learning rates, run the following command in the terminals:
  ```
  python main_q1.py --model lenet --lr 0.01
  python main_q1.py --model lenet --lr 0.001
  python main_q1.py --model lenet --lr 0.0001
  ```

# Results and Discussion

* For the default-hyperparameter experiment:
  * What is the trend for training and validation loss?
  * Why does this happen? You may want to look at [overfitting](https://en.wikipedia.org/wiki/Overfitting). 
* For the hyperparameter-tuning experiments
  * Do smaller or larger batch sizes lead to lower validation loss? Why?
  * Do smaller or larger learning rates lead to lower validation loss? Why?
* Except for batch size and learning rate, list at least three ways that could improve validation performance.

# (Optional: not for grading) AlexNet Experiments

* Complete the class AlexNet(), and run training and testing. 
* Does AlexNet perform better than LeNet? Why?

# References
* LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
* Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
* Paszke, Adam, et al. "Pytorch: An imperative style, high-performance deep learning library." Advances in neural information processing systems 32 (2019).
