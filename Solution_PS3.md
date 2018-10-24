---
title: "Data Analytics and Machine Learning: HW 3"
Author: 
date: "October 16, 2018"
output: 
  html_document: 
    keep_md: yes
---

####Group: Mengshu (Alice) Fu, Hang Hang, Rongrong (Rebecca) Nie, Somesh Srivastava, Tianwei (David) Sun

#


```r
library(knitr)
opts_chunk$set(echo = TRUE)
opts_chunk$set(tidy.opts=list(width.cutoff=60),tidy=TRUE)
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(data.table)) install.packages("data.table")
if (!require(foreign)) install.packages("foreign")

setwd("D:/MFE/Curriculum/Fall 2018/Machine Learning/PS 3")
```



## Question 1.a.(i)

Drop all rows where "loan_status" is not equal to either "Fully Paid" or "Charged Off."
Define the new variable Default as 1 (or TRUE) if "loan_status" is equal to "Charged
Off", and 0 (or FALSE) otherwise.
\newline 


```r
LendingClub <- as.data.table(read.dta("LendingClub_LoanStats3a_v12.dta"))
LendingClub[, `:=`(Default, ifelse(loan_status == "Charged Off", 
    1, ifelse(loan_status == "Fully Paid", 0, NaN)))]
LendingClub <- LendingClub[!is.na(Default)]
```

## Question 1.a.(ii)

Report the average default rate in the sample (number of defaults divided by total number of loans)
\newline


```r
DefaultRate <- sum(LendingClub$Default)/nrow(LendingClub)

sprintf("Default Rate: %f", DefaultRate)
```

```
## [1] "Default Rate: 0.143535"
```

## Question 1.b.(i)

Using the glm function, run a logistic regression of the Default variable on the grade.
Report and explain the regression output. I.e., what is the interpretation of the
coefficients? Do the numbers 'make sense'.



```r
out = glm(formula = Default ~ grade, family = "binomial", data = LendingClub)
summary(out)
```

```
## 
## Call:
## glm(formula = Default ~ grade, family = "binomial", data = LendingClub)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -0.8827  -0.6077  -0.5053  -0.3511   2.3736  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -2.75542    0.04203  -65.56   <2e-16 ***
## gradeB       0.76143    0.05061   15.04   <2e-16 ***
## gradeC       1.15967    0.05153   22.50   <2e-16 ***
## gradeD       1.46001    0.05381   27.13   <2e-16 ***
## gradeE       1.69834    0.06030   28.17   <2e-16 ***
## gradeF       1.97319    0.07933   24.87   <2e-16 ***
## gradeG       2.01395    0.12800   15.73   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 32423  on 39411  degrees of freedom
## Residual deviance: 30914  on 39405  degrees of freedom
## AIC: 30928
## 
## Number of Fisher Scoring iterations: 5
```


Best borrower grade is "A" and worst borrower grade is "G". The coefficients of regression show a linear trend - Higher for G and low for A (intercept) which depicts that poor grade borrowers are more likely to default.

## Question 1.b.(ii)

Construct and report a test of whether the model performs better than the null model
where only "beta0", and no conditioning information, is present in the logistic model.


```r
test_stat <- out$null.deviance - out$deviance
k = out$df.null - out$df.residual
pvalue_chisq <- 1 - pchisq(test_stat, df = k)

report <- matrix(c(test_stat, k, pvalue_chisq), 3, 1)
row.names(report) <- c("tstat", "degree of freedom", "p-value")
colnames(report) <- ""
kable(report, caption = "Losistic model vs null model")
```



Table: Losistic model vs null model

                             
------------------  ---------
tstat                1508.097
degree of freedom       6.000
p-value                 0.000
------------------  ---------




## Question 1.b.(iii) 

Construct the lift table and the ROC curve for this model. Explain the interpretion of
the numbers in the lift table and the lines and axis in the ROC curve. Does the model
perform better than a random guess?
\newline



```r
phat <- predict(out, type = "response")
phat <- jitter(phat, amount = 0)
deciles <- cut(phat, breaks = quantile(phat, probs = c(seq(from = 0, 
    to = 1, by = 0.1))), include.lowest = TRUE)
deciles <- as.numeric(deciles)
df <- data.frame(deciles = deciles, phat = phat, default = LendingClub$Default)
lift <- aggregate(df, by = list(deciles), FUN = "mean", data = df)
lift <- lift[, c(2, 4)]
lift[, 3] <- lift[, 2]/mean(LendingClub$Default)
names(lift) <- c("decile", "Mean Response", "Lift Factor")
print(lift)
```

```
##    decile Mean Response Lift Factor
## 1       1    0.06519533   0.4542122
## 2       2    0.05379345   0.3747760
## 3       3    0.08551129   0.5957523
## 4       4    0.12509515   0.8715309
## 5       5    0.11875159   0.8273356
## 6       6    0.13702106   0.9546180
## 7       7    0.16493276   1.1490772
## 8       8    0.19385943   1.3506077
## 9       9    0.21238264   1.4796579
## 10     10    0.27879249   1.9423316
```

```r
simple_roc <- function(labels, predicted_value) {
    labels <- labels[order(predicted_value, decreasing = TRUE)]
    data.frame(TPR = cumsum(labels)/sum(labels), FPR = cumsum(!labels)/sum(!labels), 
        labels)
}

glm_simple_roc <- simple_roc(LendingClub$Default == "1", phat)
TPR <- glm_simple_roc$TPR
FPR <- glm_simple_roc$FPR

# plot the corresponding ROC curve
q <- qplot(FPR, TPR, xlab = "FPR", ylab = "TPR", col = I("blue"), 
    main = "ROC Curve for Logistic Regression Default Model", 
    size = I(0.75))
# add straight 45 degree line from 0 to 1
q + geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), size = I(1)) + 
    theme_bw()
```

![](Solution_PS3_files/figure-html/unnamed-chunk-5-1.png)<!-- -->


In the Lift table, mean response is small for 1st decile and large for 10th decile and shows almost even increase barring
minor exception at 5th percentile. Which says model is good fit.

In the ROC curve, x-axis is False positive rate and y-axis is true positive rate.
The fitted curve is above 45 degree line which means, model is better than random guess.


## Question 1.b.(iv) 

Assume that each loan is for $100, and that you make a $1 profit if there is no default,
but lose $10 if there is a default (both given in present value terms to keep things
easy). Using data from the ROC curve (True Positive Rate and False Positive Rate)
along with the average rate of default (total number of defaults divided by total
number of loans), what is the cutoff default probability you should use as your
decision criterion to maximize profits? Plot the corresponding point on the ROC curve.


## Question 1.c.(i) 

First, consider a logistic regression model that uses only loan amount (loan_amnt) and
annual income (annual_inc) as explantory variables. Report the regression results.
Show the lift table, comparing to the 'grade'-model from a. Plot the ROC curves of both
the 'grade'-model and the altnerative model. Which model performs better?



```r
out_amt_inc = glm(formula = Default ~ loan_amnt + annual_inc, 
    family = "binomial", data = LendingClub)
summary(out_amt_inc)
```

```
## 
## Call:
## glm(formula = Default ~ loan_amnt + annual_inc, family = "binomial", 
##     data = LendingClub)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -0.8525  -0.5832  -0.5393  -0.4766   4.4804  
## 
## Coefficients:
##               Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -1.725e+00  3.213e-02  -53.71   <2e-16 ***
## loan_amnt    3.484e-05  2.081e-06   16.74   <2e-16 ***
## annual_inc  -7.089e-06  4.663e-07  -15.20   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 32423  on 39411  degrees of freedom
## Residual deviance: 32027  on 39409  degrees of freedom
## AIC: 32033
## 
## Number of Fisher Scoring iterations: 5
```

```r
phat_amt_inc <- predict(out_amt_inc, type = "response")

phat_amt_inc <- jitter(phat_amt_inc, amount = 0)
deciles <- cut(phat_amt_inc, breaks = quantile(phat_amt_inc, 
    probs = c(seq(from = 0, to = 1, by = 0.1))), include.lowest = TRUE)
deciles <- as.numeric(deciles)
df <- data.frame(deciles = deciles, phat = phat_amt_inc, default = LendingClub$Default)
lift <- aggregate(df, by = list(deciles), FUN = "mean", data = df)
lift <- lift[, c(2, 4)]
lift[, 3] <- lift[, 2]/mean(LendingClub$Default)
names(lift) <- c("decile", "Mean Response", "Lift Factor")
print(lift)
```

```
##    decile Mean Response Lift Factor
## 1       1    0.09005581   0.6274137
## 2       2    0.09718346   0.6770717
## 3       3    0.11900533   0.8291034
## 4       4    0.12484141   0.8697631
## 5       5    0.13422989   0.9351721
## 6       6    0.14311089   0.9970455
## 7       7    0.14843948   1.0341695
## 8       8    0.15833545   1.1031141
## 9       9    0.19360568   1.3488399
## 10     10    0.22653475   1.5782549
```

```r
glm_simple_roc_amt_inc <- simple_roc(LendingClub$Default == "1", 
    phat_amt_inc)
TPR_amt_inc <- glm_simple_roc_amt_inc$TPR
FPR_amt_inc <- glm_simple_roc_amt_inc$FPR

# plot the corresponding ROC curve plot the corresponding ROC
# curve
q <- ggplot() + geom_line(data = glm_simple_roc, aes(x = FPR, 
    y = TPR, colour = "Default"))
# add straight 45 degree line from 0 to 1
q <- q + geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1, colour = "Random"), 
    size = I(1))

q <- q + geom_line(data = glm_simple_roc_amt_inc, aes(x = FPR_amt_inc, 
    y = TPR_amt_inc, colour = "with_Amt_income"))
q <- q + labs(x = "FPR", y = "TPR", colour = "Labels", title = "ROC Curve for Logistic Reg Default vs restricted (amt, Income) Model")
q
```

![](Solution_PS3_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

Based on the ROC curve, Default model is better than Restricted model.


## Question 1.c.(ii) 

Now, include also information from the loan itself. In particular, include the maturity
of the loan (term) and the interest rate (int_rate) in the logistic regression. Report the
output. How does R handle the term-variable? In particular, what is the interpretation
of the regression coefficient? Again show the lift table and ROC curve relative to the
original 'grade' model. Now, which model is better? What is the likely explanation for
why this new model performs better/worse?


```r
out_4info = glm(formula = Default ~ loan_amnt + annual_inc + 
    term + int_rate, family = "binomial", data = LendingClub)
summary(out_4info)
```

```
## 
## Call:
## glm(formula = Default ~ loan_amnt + annual_inc + term + int_rate, 
##     family = "binomial", data = LendingClub)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.2520  -0.5868  -0.4694  -0.3598   4.1684  
## 
## Coefficients:
##                  Estimate Std. Error z value Pr(>|z|)    
## (Intercept)    -3.266e+00  6.055e-02 -53.942   <2e-16 ***
## loan_amnt       1.176e-06  2.311e-06   0.509    0.611    
## annual_inc     -6.117e-06  4.643e-07 -13.173   <2e-16 ***
## term 60 months  4.538e-01  3.564e-02  12.732   <2e-16 ***
## int_rate        1.349e+01  4.560e-01  29.575   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 32423  on 39411  degrees of freedom
## Residual deviance: 30418  on 39407  degrees of freedom
## AIC: 30428
## 
## Number of Fisher Scoring iterations: 5
```

```r
phat_4info <- predict(out_4info, type = "response")

phat_4info <- jitter(phat_4info, amount = 0)
deciles <- cut(phat_4info, breaks = quantile(phat_4info, probs = c(seq(from = 0, 
    to = 1, by = 0.1))), include.lowest = TRUE)
deciles <- as.numeric(deciles)
df <- data.frame(deciles = deciles, phat = phat_4info, default = LendingClub$Default)
lift <- aggregate(df, by = list(deciles), FUN = "mean", data = df)
lift <- lift[, c(2, 4)]
lift[, 3] <- lift[, 2]/mean(LendingClub$Default)
names(lift) <- c("decile", "Mean Response", "Lift Factor")
print(lift)
```

```
##    decile Mean Response Lift Factor
## 1       1    0.03982750   0.2774759
## 2       2    0.06140573   0.4278103
## 3       3    0.07789901   0.5427180
## 4       4    0.10682568   0.7442485
## 5       5    0.11164679   0.7778369
## 6       6    0.14209591   0.9899742
## 7       7    0.15909668   1.1084176
## 8       8    0.18624715   1.2975734
## 9       9    0.23927937   1.6670459
## 10     10    0.31100964   2.1667866
```

```r
glm_simple_roc_4info <- simple_roc(LendingClub$Default == "1", 
    phat_4info)
TPR_4info <- glm_simple_roc_4info$TPR
FPR_4info <- glm_simple_roc_4info$FPR

# plot the corresponding ROC curve plot the corresponding ROC
# curve
q <- ggplot() + geom_line(data = glm_simple_roc, aes(x = FPR, 
    y = TPR, colour = "Default"))
# add straight 45 degree line from 0 to 1
q <- q + geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1, colour = "Random"), 
    size = I(1))

q <- q + geom_line(data = glm_simple_roc_amt_inc, aes(x = FPR_amt_inc, 
    y = TPR_amt_inc, colour = "with_Amt_income"))
q <- q + geom_line(data = glm_simple_roc_4info, aes(x = FPR_4info, 
    y = TPR_4info, colour = "with_Amt_income_term_rate"))
q <- q + labs(x = "FPR", y = "TPR", colour = "Labels", title = "ROC Curve for Logistic Reg Default vs restricted I vs Restricted II Model")
q
```

![](Solution_PS3_files/figure-html/unnamed-chunk-7-1.png)<!-- -->


## Question 1.c.(iii) 

Create the squared of the interest rate and add this variable to the last model. Is the
coefficient on this variable significant? Please give an intuition for what the
coefficients on both int_rate and its squared value imply for the relationship between
defaults and the interest rate.


