import numpy as np
from math import *
from tqdm import tqdm

class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        """
        Parameters:
        - penalty: str, "l1" or "l2". Determines the regularization to be used.
        - gamma: float, regularization coefficient. Used in conjunction with 'penalty'.
        - fit_intercept: bool, whether to add an intercept (bias) term.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        ################################################################################
        # TODO:                                                                        #
        # Implement the sigmoid function.
        ################################################################################
        
        return 1/(1+exp(-x))
        
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=10000):
        """
        Fit the regression coefficients via gradient descent or other methods 
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        - y: numpy array of shape (n_samples,), target data.
        - lr: float, learning rate for gradient descent.
        - tol: float, tolerance to decide convergence of gradient descent.
        - max_iter: int, maximum number of iterations for gradient descent.
        Returns:
        - losses: list, a list of loss values at each iteration.        
        """
        # If fit_intercept is True, add an intercept column
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Initialize coefficients
        self.coef_ = np.zeros(X.shape[1])
        
        # List to store loss values at each iteration
        losses = []

        ################################################################################
        # TODO:                                                                        #
        # Implement gradient descent with optional regularization.
        # 1. Compute the gradient 
        # 2. Apply the update rule
        # 3. Check for convergence
        ################################################################################
        for j in tqdm(range(1,int(max_iter))):
            beta=self.coef_
            loss=0
            l_prime=np.zeros(X.shape[1])
            P=np.zeros(X.shape[1])
            for i in range(X.shape[0]):
                z=np.dot(beta,X[i])
                p_1=exp(z)*self.sigmoid(-z)
                l_prime=l_prime-X[i]*(y[i]-p_1)# 一阶导
            for i in range(X.shape[1]):
                if beta[i]!=0:
                    P[i]=1/(2*beta[i])
            D=np.diag(P)
            if self.penalty=='l1':
                l_prime=l_prime+self.gamma*2*np.dot(D,beta)
            if self.penalty=='l2':
                l_prime=l_prime+self.gamma*2*np.linalg.norm(beta,ord=2)*2*np.dot(D,beta)
            if np.linalg.norm(l_prime,ord=2) < tol:
                break
            self.coef_=beta-lr*l_prime# 更新参数
            for i in range(X.shape[0]):
                z=np.dot(beta,X[i])
                if self.penalty=='l1':
                    loss=loss-y[i]*z+log(1+exp(z))+self.gamma*np.linalg.norm(self.coef_,ord=1)# 计算损失
                if self.penalty=='l2':
                    loss=loss-y[i]*z+log(1+exp(z))+self.gamma*np.linalg.norm(self.coef_,ord=2)*np.linalg.norm(self.coef_,ord=2)
            losses.append(loss)
        
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return losses

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        
        Returns:
        - probs: numpy array of shape (n_samples,), prediction probabilities.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Compute the linear combination of inputs and weights
        linear_output = np.dot(X, self.coef_)
        probs=np.zeros(X.shape[0])
        
        ################################################################################
        # TODO:                                                                        #
        # Task3: Apply the sigmoid function to compute prediction probabilities.
        ################################################################################

        for i in range(X.shape[0]):
            key=self.sigmoid(linear_output[i])
            if key >= 0.5:
                probs[i]=1
            else:
                probs[i]=0
        return probs
        
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
