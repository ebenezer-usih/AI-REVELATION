#POLYNOMIAL REGRESSION
# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split=sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set=subset(dataset,split==TRUE)
# test_set=subset(dataset,split==FALSE)

#feature scaling
# training_set[,2:3]=scale(training_set[,2:3])
# test_set[,2:3]=scale(test_set[,2:3])

#FITTING LINEAR REGRESSION TO DATASET
lin_reg = lm(formula = Salary ~ Level,
             data = dataset)

#FITTING POLYNOMIAL REGRESSION TO DATASET
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

#VISUALISING LINEAR REGRESSION RESULTS
ggplot() +
  geom_point(aes(dataset$Level, dataset$Salary), 
             color = 'red') + 
  geom_line(aes(dataset$Level, predict(lin_reg, newdata = dataset)), 
                color = 'blue') + 
  ggtitle('SALARY VS LEVEL(LINEAR REGRESSION)') + 
  xlab('LEVEL') + 
  ylab('SALARY')

#VISUALISING POLYNOMIAL REGRESSION RESULTS
ggplot() +
  geom_point(aes(dataset$Level, dataset$Salary), 
             color = 'red') + 
  geom_line(aes(dataset$Level, predict(poly_reg, newdata = dataset)), 
            color = 'blue') + 
  ggtitle('SALARY VS LEVEL(POLYNOMIAL REGRESSION)') + 
  xlab('LEVEL') + 
  ylab('SALARY')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg,
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

#PREDICTING RESULT WITH LINEAR REGRESSION
y_pred = predict(lin_reg, data.frame(Level=6.5))

#PREDICTING RESULT WITH POLYNOMIAL REGRESSION
y_pred = predict(poly_reg, data.frame(Level=6.5,
                                     Level2=6.5^2,
                                     Level3=6.5^3,
                                     Level4=6.5^4))