# IITGN

This project is part of the screening process to work on AI/ML projects under guidance from professors at IIT, Gandhinagar. The process had the folllowing 4 questions which had to be completed over a course of roughly 5 days:

1. Bivariate Normal Distribution Plot :white_check_mark: - Plot the bivariate normal as shown in the image below: 

    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/MultivariateNormal.png/330px-MultivariateNormal.png">.

    The plot must include intractive sliders using which users can vary the parameters for the distribution. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nitish1295/IITGN.git/master?filepath=Bivariate_Normal_Distribution.ipynb)
    
2. Sampling from MVN :white_check_mark: - Write a sampling method from scratch, the sampling method must produce samples from a Multivariate Normal Distribution based on the randomly generated mean and covariance matrix. The method must work for any number of dimensions and you are not allowed to use any default libraries to generate a normal distribution.[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_omE5a9EWDLLlHDQLgZdL65earF23Qu3?usp=sharing)

3. Implement a Neural Network from Scratch :negative_squared_cross_mark: - Write a neural network from scratch which works with MNIST data set. The network must optimize itself using gradient descent which is written from scratch. Finally plot graphs for train and test and evaluate the model based on different classification metrics

4. Implement Bayesian Regression from Scratch :negative_squared_cross_mark: - Write Bayesian Linear Regression from Scratch and plot the learned predictive mean and 2 standard deviations around it. Use your own 1-D dataset with noise.
---

## Constraints:
- You are only allowed to use <a href="https://jax.readthedocs.io/en/latest/notebooks/quickstart.html">JAX</a> unless mentioned otherwise.
- Your code must adhere to PEP8

---

## Learning Outcomes and Challenges:
1. Bivariate Normal Distribution Plot : I had never used JAX before so starting with it was bit of a challenge initially, but once I learned that JAX worked pretty much like numpy in some aspects, it was smooth sailing :boat:. Although I did get caught up with <a href="https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html"> immutable arrays in JAX </a> and ipywidgets functionalities at a later point. From a mathematical standpoint I had to uderstand positive definite matrices and how to generate them so that the JAX multivariate normal function can return valid values for a given covariance matrix(FYI - I understood the complete idea behind this once I completed the second question below)

2. Sampling from MVN : In this task I did understand that you can use Weak Law of Large numbers to generate standard normals but did not understand how that will work with a random mean and covariance matrix in a Mutlivariate Setting. After reading about for a while I discovered a method which uses Cholesky decompositon to create samples from mutivariate normal.

3. Implement a Neural Network from Scratch : I am still working on implementing this neural network, but till now I have only implemented forward and backward pass for the dense layers, it is pretty rudimentry hence I havent included it here. This task is slightly challenging but exciting at the same time and I plan to continue working on this, since I haven't implemented a Neural Net from scratch before.
---

## Technologies/ Libraries:
- Python Libraries
    - JAX
    - Plotly
    - matplotlib
