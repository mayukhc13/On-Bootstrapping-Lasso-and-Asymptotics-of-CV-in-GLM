library(glmnet) #for lasso
library(writexl) # to store sequence of values of \lambda_hat_cv/\sqrt(n)

n=500 ## SAMPLE SIZE

p=7 ## number of parameter components
p_0=4 #non zero components



beta=vector()

for(j in 1:p)
{
  if(j<=p_0)
  {
    beta[j]= 0.5*j*((-1)^j)
  }
  else{
    beta[j]=0
  }
}
beta  #true parameter choice



library(MASS) #for multivariate normal sampling
mu= rep(0,p)  ## mean vector with all zero components
mu
cov_matrix = matrix(NA,p,p)
for(i in 1:p)
{
  for(j in 1:p)
  {
    if(i==j)
    {
      cov_matrix[i,j] = 1
    }
    else
    {
      cov_matrix[i,j]= 0.3^(abs(i-j))
    }
  }
}
cov_matrix #covariance matrix


x = mvrnorm(n,mu,cov_matrix) # matrix, drawn from multivariate normal with above mean and covariance structure
x

uv= as.matrix(rep(1,n))
uv
x_int1=cbind(uv,x)
x_int1
x_int=as.matrix(x_int1)
x_int # earlier design matrix with first column all with 1

z = scale(x,T,F) ## design matrix without intercept column, is just centered which will be used as actual design matrix throuhout.
z
z_int = scale(x_int,T,F)
z_int


P = vector()
for (i in 1:n)
{
  P[i]= exp(z[i,]%*%beta)/(1+exp(z[i,]%*%beta))  ## from logit link relation for success prob.
}
P # success prob vector of n components with which n bernoulli response variables will be drawn.

simulation=500 # entire iteration of data to be done
lambda_matrix = matrix(NA, simulation,1)  # array to store 500 optimal choices of \lambda_hat_cv/\sqrt(n)
for(m in 1:simulation)
{
  y= rbinom(n,size = 1,prob = P)
  y  # Bernoulli response vector of n components. 
  
  y_final = data.frame(y)
  x_final = data.frame(z)
  logistic_data = cbind(y,z)
  logistic_data_final = data.frame(logistic_data)
  
  
  
  lambda_sequence = 10^seq(2, -2, length.out = 100) 
  lambda_sequence 
  CV = cv.glmnet(x=z_int,y=logistic_data_final$y,family= "binomial",alpha=1,lambda = lambda_sequence,type.measure = "deviance",intercept=F,nfolds = 10)
  CV #cross validation for optimal lambda in logistic regression.
  lambda_opt = CV$lambda.min
  lambda_matrix[m,1]=lambda_opt/sqrt(n) ## optimal scaled lambda_hat_cv sequence
  
}

lambda_matrix
lambda_matrix1=as.data.frame(lambda_matrix)
lambda_matrix1
write_xlsx(lambda_matrix1,path = "E://first work revised//logistic//lambda_hat.xlsx") # change this path as per your own directory
sim=seq(1,simulation,by=1)
sim

hist(lambda_matrix,xlab = "lambda_hat/sqrt_n",col = "red",border = "black",main = "Histogram of lambda_hat upon sqrt(n)")
