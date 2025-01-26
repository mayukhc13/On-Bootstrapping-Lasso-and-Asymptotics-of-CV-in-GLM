# On-Bootstrapping-Lasso-and-Asymptotics-of-CV-in-GLM
This repository contains reproducible codes regarding the simulation study of the paper. It also contains the real life clinical data analysis.


# Simulation Study Synopsis:

## The simulation study is attributed to the cases of Logistic Regression, Gamma Regression and Linear Regression Models. 


## Change (n,B)=(sample size, Bootstrap iteration count) as required. 
Our choices for n include: (50,100,150,300,500) and B=n.

## Change (p,p_0)=(dimension of beta parameter,first p_0 many among p are non-zero components) as required.
Our choices for (p,p_0) include: {(5,2),(7,4),(8,3)}

## Now for each of (n,p,p_0), the design matrix is once and initially generated from some structure outside the loop and kept fixed throughout the entire simulation. 

## The entire data set is generated (simulation=500) times. Change this as needed.

## Now for thresholded lasso estimator part (named as: beta_tilde in .R files), the pivotal thing is the choice of a_n (thresholding level)
Choice of a_n is given as: n^(-c) and c varies as: (0.0015, 1/6, 1/5, 1/4, 1/3, 0.485). 
Change these choices as needed.


## K-fold Cross-Validation Part:
Number of folds K=10 is fixed (nfolds=10 command in cv.glmnet)
each fold size m=n/K changes as per that and is utilized in the back end of cv.glmnet command. 


## Required Packages
CVXR package for convex optimization is used.
glmnet package is used for elastic net in Logistic and Linear Regression. 
h2o package is used for elastic net in Gamma regression.



## Comparative Analysis (Insight 1:) (fixed (p,p_0), fixed a_n and varying choices of n)
We have calculated empirical coverage probabilities (for both sided and one sided) and average widths (for both sided) of nominal 90% Bootstrap percentile confidence intervals.
For fixed (p,p_0)=(7,4), a fixed a_n=n^(-1/3), we have compared the empirical probabilities over n=(50,100,150,300,500).
These procedures are performed in Logistic, Gamma and Linear Regressions.


## Comparative Analysis (Insight 2:) (fixed (p,p_0), varying a_n and varying choices of n)
We have calculated empirical coverage probabilities (for both sided and one sided) and average widths (for both sided) of nominal 90% Bootstrap percentile confidence intervals.
For fixed (p,p_0)=(7,4), varying a_n=n^(-c), with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485), we have compared the empirical probabilities over n=(50,100,150,300,500).
These procedures are performed in Logistic Regression.


## Comparative Analysis (Insight 3:) (varying (p,p_0), fixed a_n and varying choices of n)
We have calculated empirical coverage probabilities (for both sided and one sided) and average widths (for both sided) of nominal 90% Bootstrap percentile confidence intervals.
For varying (p,p_0)={(5,2),(7,4),(8,3)}, a fixed a_n=n^(-1/3), we have compared the empirical probabilities over n=(50,100,150,300,500).
These procedures are performed in Logistic Regression.


## Run these .R files as required.




# Synopsis of Real Life Clinical Data Analysis

## Attribute Information :

Link : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

## Load the excel file named dataR2.xlsx attached in this folder named "Real Data Analysis" first while reproduce the codes.

## The number of observations n is 116. 
The response variable is binary. In original data set by 1, denote "healthy control(or absence) and by 2, denote "patients(or presence of cancer)".
For ease of notations, we adjusted 1 and 2 respectively by 0 and 1 in the simulation.

## The 9 predictive factors are : {Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP.1}

## Choose B=150, a_n=n^(-1/3) and K=10

## We applied our proposed method to get important factors that involve in detection of Breast Cancer.
Here we obtain Lasso estimate of all these 9 factors and 90% Simultaneous Bootstrap Percentile Intervals.

## Finally based on these intervals and lasso estimates, we capture the important factors weighing to the presence of Breast Cancer.
This coincides with the findings of Patricio et al.(2018).




