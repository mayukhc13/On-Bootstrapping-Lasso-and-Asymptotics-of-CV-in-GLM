#set.seed(100)
library(writexl) #To fix this design matrix for reuse under (n,p,p0)
library(CVXR) # for convex optimization


n=500 ## SAMPLE SIZE
## change n accordingly

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

design1=as.data.frame(x)
design1
write_xlsx(design1,path = "M://Revised First Work//Linear Related//linear_design_500_7_4.xlsx") # change the path according to your own directory named as "linear_design_500_7_4.xlsx"


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
  P[i]= (z[i,]%*%beta)  ## from identity link relation for mean parameter of gaussian.
}
P #  mean vector of n components with which n standard Gaussian response variables will be drawn.


simulation = 500 # entire iteration of data to be done
width_matrix = matrix(NA, simulation,p) #both sided CI matrix width, (upper confidence limit-lower confidence limit) for each component
count_coverage = matrix(NA,simulation,p) # both sided empirical coverage count matrix, count is 1 if true parameter value falls within both sided limit
count_coverage_right = matrix(NA,simulation,p) #right sided empirical coverage count matrix
count_coverage_left = matrix(NA,simulation,p)  # left sided empirical coverage count matrix
count_norm_matrix = matrix(NA,simulation,1) # count coverage in terms of euclidean norm 

for(m in 1:simulation)
{
  y= vector()
  for(i in 1:n)
  {
    y[i]= rnorm(1,mean=P[i], sd= 1) 
  }
  y  # standard Gaussian response vector of n components. 
  
  
  y_final = data.frame(y)
  x_final = data.frame(z)
  linear_data = cbind(y,z)
  linear_data_final = data.frame(linear_data)
  
  
  library(glmnet) #for lasso
  lambda_sequence = 10^seq(2, -2, length.out = 100) 
  lambda_sequence #user defined choice to search for optimal cross validated penalty parameter
  CV = cv.glmnet(x=z_int,y=logistic_data_final$y,family= gaussian(link = "identity"),alpha=1,lambda = lambda_sequence,type.measure = "deviance",intercept=F,nfolds = 10)
  CV #cross validation for optimal lambda in linear regression.
  lambda_opt = CV$lambda.min
  lambda_opt ## optimal lambda to be used later in Bootstrap
  
  
  #extracting lasso coefficients
  fit_model= glmnet(x=z_int,y=logistic_data_final$y,family = gaussian(link = "identity"),alpha =1,intercept = F,penalty.factor = c(0,rep(1,p)),lambda = 0)
  fit_model  # fitting linear lasso without intercept
  beta_hat = coef(fit_model)
  beta_hat
  beta_hat1 = as.array(beta_hat)
  beta_hat2 = data.frame(beta_hat1)
  
  beta_hat_final = beta_hat2$s0
  beta_hat_final
  beta_hat_final3 = beta_hat_final[-c(1,2)]
  beta_hat_final3 #lasso estimate of true parameter
  
  
  #centered distribution of (lasso estimate-original parameter)*sqrt(n)
  stat3= sqrt(n)*(beta_hat_final3-beta) 
  stat3
  stat4= as.matrix(stat3)
  stat4
  
  # euclidean norm of above to check empirical coverage probabilities of parameter vector for 90% CI in terms of norm
  stat3_norm = norm(stat4,type = "f") 
  stat3_norm
  
  
  # Next we do, thresholding of Lasso estimator
  ##we will choose via a_n=n^(-c) for 0<c<1/2, precisely c=1/3 
 
  a_n = n^(-(1/3)) 
  beta_tilde = vector()
  for(i in 1:p)
  {
    if(abs(beta_hat_final3[i])>a_n)
    {
      beta_tilde[i]= beta_hat_final3[i]
    }
    else
    {
      beta_tilde[i]=0
    }
  }
  beta_tilde #thresholded lasso estimate
  
  
  P_tilde = vector()
  for(i in 1:n)
  {
    P_tilde[i]= (z[i,]%*%beta_tilde) ## relation from identity link
  }
  P_tilde  #mean vector at beta tilde
  
  
  B=n #bootstrap iterations to be made
  beta_bootstrap = matrix(NA,B,p) # collect B copies of bootstrap estimate of lasso estimator
  for(c in 1:B)
  {
    ## Perturbation Bootstrap Quantities Generation
    G_boot = matrix(NA,1,n)
    for (j in 1:n)
    {
      G_boot[1,j]= rexp(1,1) ## extract n iid exp(1) r.v for each stage optimization in objective function.
    }
    
    G_boot #perturbation quantities from known density Exp(1), n copies at each stage, repeat B times
    
    ## Next we do Bootstrap convex optimization to get PB-Lasso Estimator though CVXR function
    ##declaration: bootstrap estimate wrt which objective function is optimized
    v3 = CVXR::Variable(p)
    ## penalty term in the objective fn through p_norm syntax.
    penalty_boot = (p_norm(v3,1))*lambda_opt
    ## a col vector consisting of component wise product of y and G_boot. Also * does component wise product and return a vector
    zz= t(y*G_boot)
    ##  %*% returns actually a scalar product and l1 is the first component in the first term as in Bootstrap obj fn
    l1= -sum((z%*%v3)*zz) 
    ## col vector (y-Ptilde)
    zz1= as.matrix(y-P_tilde) 
    ## l2 is the second term as in Bootstrap obj fn
    l2= sum((z%*%v3)*zz1) 
    ## col vector G_boot
    zz2= t(G_boot) 
    ##l3 is the second component in the first term as in Bootstrap obj fn
    l3 = sum((((z%*%v3)^2)/2)*zz2) 
    
    
    obj_boot = penalty_boot+l1+l2+l3  
    prob_boot = Problem(Minimize(obj_boot)) ## minimisation of objective function
    resultB = solve(prob_boot)
    resultB
    
    final_boot1 = resultB$getValue(v3)
    final_boot1
    final_boot2=as.matrix(final_boot1)
    final_boot2
    
    final_boot3 = t(final_boot2)
    final_boot3 ## Minimizer Bootstrap estimator 
    
    beta_bootstrap[c,]= final_boot3 ## At c-th stage we store that bootstrap estimator of p components
  }
  beta_bootstrap ##This matrix is important to construct Bootstrap Percentile Intervals
  
  ## unit vector of B replicas of 1
  unit_vector = as.matrix(rep(1,B))
  beta_tilde_matrix = unit_vector%*%beta_tilde
  beta_tilde_matrix
  
  ## Tau boot is the matrix containing B copies of centered estimator vector sqrt(n)*(Bootstrap estimator - thresholded lasso estimator)
  tau_boot= matrix(NA,B,p)
  for(i in 1:B){
    for(j in 1:p){
      tau_boot[i,j]=sqrt(n)*(beta_bootstrap[i,j]-beta_tilde_matrix[i,j])
    }
  }
  tau_boot ## B copies of centered bootstrap-beta tilde distribution scaled with sqrt n
  
  sorted_tau_boot = apply(tau_boot,2,sort)
  sorted_tau_boot ## independent column wise sorting of tau boot for getting critical value in percentile intervals
  
  
  alpha = 0.10 #level of significance
  lower_index = as.integer(B*(alpha/2)) #both sided
  upper_index = 1+ as.integer(B*(1-(alpha/2))) #both sided
  index_right = as.integer(B*alpha) #right sided
  index_left = 1+ as.integer(B*(1-alpha)) #left sided
  
  lower_index
  upper_index
  index_left
  index_right
  
  tau_boot_norm = matrix(NA,B,1) #euclidean norm of each row of tau boot
  for(i in 1:B){
    tau_boot_norm[i,1]= norm(as.matrix(tau_boot[i,]),type = "f")
  }
  tau_boot_norm
  
  sorted_tau_boot_norm = apply(tau_boot_norm,2,sort) #sorted
  sorted_tau_boot_norm
  
  if(stat3_norm<=sorted_tau_boot_norm[index_left,1]){
    count_norm = 1
  }else
  {
    count_norm = 0
    
  }
  count_norm_matrix[m,1]=count_norm #count if stat3norm is less than sorted tauboot norm at 0.90 level
  
  critical_value_matrix = matrix(NA,2,p)
  for(j in 1:p)
  {
    critical_value_matrix[1,j]= -(sorted_tau_boot[upper_index,j])/sqrt(n)
    critical_value_matrix[2,j]= -(sorted_tau_boot[lower_index,j])/sqrt(n)
  }
  critical_value_matrix #both sided
  
  critical_value_matrix_right = matrix(NA,1,p)
  right_confidence_limit_matrix = matrix(NA,1,p)
  for(j in 1:p){
    critical_value_matrix_right[1,j] = -(sorted_tau_boot[index_right,j])/sqrt(n)
    right_confidence_limit_matrix[1,j]= beta_hat_final3[j]+critical_value_matrix_right[1,j]
  }
  critical_value_matrix_right
  right_confidence_limit_matrix #right sided
  
  critical_value_matrix_left = matrix(NA,1,p)
  left_confidence_limit_matrix = matrix(NA,1,p)
  for(j in 1:p){
    critical_value_matrix_left[1,j] = -(sorted_tau_boot[index_left,j])/sqrt(n)
    left_confidence_limit_matrix[1,j]= beta_hat_final3[j]+critical_value_matrix_left[1,j]
  }
  critical_value_matrix_left
  left_confidence_limit_matrix #left sided
  
  
  lower_confidence_limit_matrix = matrix(NA,1,p)
  upper_confidence_limit_matrix = matrix(NA,1,p)
  for(j in 1:p)
  {
    lower_confidence_limit_matrix[1,j]= beta_hat_final3[j]+critical_value_matrix[1,j]
    upper_confidence_limit_matrix[1,j]= beta_hat_final3[j]+critical_value_matrix[2,j]
  }
  lower_confidence_limit_matrix
  upper_confidence_limit_matrix #both sided
  
  
  count_beta_right = matrix(NA,1,p)
  for(j in 1:p){
    if(beta[j]<=right_confidence_limit_matrix[1,j]){
      count_beta_right[1,j]=1
    }
    else{
      count_beta_right[1,j]=0
    }
  }
  count_beta_right
  
  count_beta_left = matrix(NA,1,p)
  for(j in 1:p){
    if(beta[j]>=left_confidence_limit_matrix[1,j]){
      count_beta_left[1,j]=1
    }
    else{
      count_beta_left[1,j]=0
    }
  }
  count_beta_left
  
  count_beta = matrix(NA,1,p)
  for(j in 1:p)
  {
    if(beta[j]>= lower_confidence_limit_matrix[1,j] && beta[j]<= upper_confidence_limit_matrix[1,j])
    {
      count_beta[1,j]= 1
    }
    else
    {
      count_beta[1,j]= 0
    }
  }
  count_beta
  
  
  width_matrix[m,] = upper_confidence_limit_matrix[1,]-lower_confidence_limit_matrix[1,]
  count_coverage[m,] = count_beta[1,]
  count_coverage_right[m,]=count_beta_right[1,]
  count_coverage_left[m,]= count_beta_left[1,]
}

width_matrix
count_coverage
count_coverage_right
count_coverage_left
count_norm_matrix


average_width_matrix = matrix(NA,1,p)
for(j in 1:p)
{
  average_width_matrix[1,j] = mean(width_matrix[,j])
}
average_width_matrix


coverage_probability_right = matrix(NA,1,p)
for(j in 1:p){
  coverage_probability_right[1,j]= mean(count_coverage_right[,j])
}
coverage_probability_right ## Right sided empirical coverage probability of right sided 90% CI

coverage_probability_left = matrix(NA,1,p)
for(j in 1:p){
  coverage_probability_left[1,j]= mean(count_coverage_left[,j])
}
coverage_probability_left ## Left sided empirical coverage probability of left sided 90% CI


coverage_probability = matrix(NA, nrow=1, ncol= p)
for(j in 1:p)
{
  coverage_probability[1,j]= mean(count_coverage[,j])
}
coverage_probability ## both sided empirical coverage probability of 90% CI

count_norm_coverage_probability = mean(count_norm_matrix[,1])
count_norm_coverage_probability

