# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[,1:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split=sample.split(dataset$Salary, SplitRatio = 2/3)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#feature scaling
# training_set[,2:3]=scale(training_set[,2:3])
# test_set[,2:3]=scale(test_set[,2:3])

#FITTING LINEAR REGRESSION TO TRAINING SET
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

#PREDICTING TEST SET RESULTS
y_pred = predict(regressor, newdata = test_set)

#VISUALISING TRAINING SET RESULT
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('SALARY VS EXPERIENCE(TRAINING SET)')+
  xlab('YEARS OF EXPERIENCE') + 
  ylab('SALARY')

#VISUALISING TEST SET RESULT
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('SALARY VS EXPERIENCE(TEST SET)')+
  xlab('YEARS OF EXPERIENCE') + 
  ylab('SALARY')
 