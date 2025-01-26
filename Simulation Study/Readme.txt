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





#### Name and Descriptions of Each .xlsx (or .csv) Files:
1. logit_design_50_7_4.xlsx : Utilise this as fixed design matrix for (n,p,p_0)=(50,7,4) when we vary a_n=n^(-c), with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485).
2. logit_design_100_7_4.xlsx : Utilise this as fixed design matrix for (n,p,p_0)=(100,7,4) when we vary a_n=n^(-c), with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485).
3. logit_design_150_7_4.xlsx : Utilise this as fixed design matrix for (n,p,p_0)=(150,7,4) when we vary a_n=n^(-c), with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485).
4. logit_design_300_7_4.xlsx : Utilise this as fixed design matrix for (n,p,p_0)=(300,7,4) when we vary a_n=n^(-c), with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485).
5. logit_design_500_7_4.xlsx : Utilise this as fixed design matrix for (n,p,p_0)=(500,7,4) when we vary a_n=n^(-c), with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485).




#### Name and Description of the .R files:

## Regression Type: Logistic

(A) For Comparative Analysis (Insight 2)

(i) logistic revised_50_7_4.R : For n=50, p=7, p_0=4 and a_n=n^(-c) with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485)
     To utilise same design matrix for different a_n, just recall "logit_design_50_7_4.xlsx" in the code instead of generating it through mvrnorm syntax.

(ii) logistic revised_100_7_4.R : For n=100, p=7, p_0=4 and a_n=n^(-c) with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485)
     To utilise same design matrix for different a_n, just recall "logit_design_100_7_4.xlsx" in the code instead of generating it through mvrnorm syntax.

(iii) logistic revised_150_7_4.R : For n=150, p=7, p_0=4 and a_n=n^(-c) with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485)
     To utilise same design matrix for different a_n, just recall "logit_design_150_7_4.xlsx" in the code instead of generating it through mvrnorm syntax.

(iv)  logistic revised_300_7_4.R : For n=300, p=7, p_0=4 and a_n=n^(-c) with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485)
     To utilise same design matrix for different a_n, just recall "logit_design_300_7_4.xlsx" in the code instead of generating it through mvrnorm syntax.

(v)  logistic revised_500_7_4.R : For n=500, p=7, p_0=4 and a_n=n^(-c) with c=(0.0015, 1/6, 1/5, 1/4, 1/3, 0.485)
     To utilise same design matrix for different a_n, just recall "logit_design_500_7_4.xlsx" in the code instead of generating it through mvrnorm syntax.


(B) For Comparative Analysis (Insight 3)

(i) logistic revised_50_5_2.R : For n=50, p=5, p_0=2 and a_n=n^(-1/3)
    logistic revised_50_7_4.R : For n=50, p=7, p_0=4 and a_n=n^(-1/3)
    logistic revised_50_8_3.R : For n=50, p=8, p_0=3 and a_n=n^(-1/3)

(ii) logistic revised_100_5_2.R : For n=100, p=5, p_0=2 and a_n=n^(-1/3)
     logistic revised_100_7_4.R : For n=100, p=7, p_0=4 and a_n=n^(-1/3)
     logistic revised_100_8_3.R : For n=100, p=8, p_0=3 and a_n=n^(-1/3)

(iii) logistic revised_150_5_2.R : For n=150, p=5, p_0=2 and a_n=n^(-1/3)
      logistic revised_150_7_4.R : For n=150, p=7, p_0=4 and a_n=n^(-1/3)
      logistic revised_150_8_3.R : For n=150, p=8, p_0=3 and a_n=n^(-1/3)

(iv) logistic revised_300_5_2.R : For n=300, p=5, p_0=2 and a_n=n^(-1/3)
     logistic revised_300_7_4.R : For n=300, p=7, p_0=4 and a_n=n^(-1/3)
     logistic revised_300_8_3.R : For n=300, p=8, p_0=3 and a_n=n^(-1/3)

(v) logistic revised_500_5_2.R : For n=500, p=5, p_0=2 and a_n=n^(-1/3)
    logistic revised_500_7_4.R : For n=500, p=7, p_0=4 and a_n=n^(-1/3)
    logistic revised_500_8_3.R : For n=500, p=8, p_0=3 and a_n=n^(-1/3)


(C) For Comparative Analysis (Insight 1)

Follow Case (A) with a_n=n^(-1/3).




## Regression Type: Gamma (Only Comparative Analysis Insight 1 is produced)

(i)  gamma revised_50_7_4.R : For n=50, p=7, p_0=4 and a_n=n^(-1/3)
   
(ii) gamma revised_100_7_4.R : For n=100, p=7, p_0=4 and a_n=n^(-1/3)
     
(iii) gamma revised_150_7_4.R : For n=150, p=7, p_0=4 and a_n=n^(-1/3)
     
(iv) gamma revised_300_7_4.R : For n=300, p=7, p_0=4 and a_n=n^(-1/3)
     
(v)  gamma revised_500_7_4.R : For n=500, p=7, p_0=4 and a_n=n^(-1/3)
    




## Regression Type: Linear (Only Comparative Analysis Insight 1 is produced)

(i)  linear revised_50_7_4.R : For n=50, p=7, p_0=4 and a_n=n^(-1/3)
   
(ii) linear revised_100_7_4.R : For n=100, p=7, p_0=4 and a_n=n^(-1/3)
     
(iii) linear revised_150_7_4.R : For n=150, p=7, p_0=4 and a_n=n^(-1/3)
     
(iv) linear revised_300_7_4.R : For n=300, p=7, p_0=4 and a_n=n^(-1/3)
     
(v)  linear revised_500_7_4.R : For n=500, p=7, p_0=4 and a_n=n^(-1/3)
    


