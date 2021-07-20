#APRIORI

#DATA PREPROCESSING
install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
itemFrequencyPlot(dataset, topN = 10)

#TRAINING APRIORI ON THE DATASET
rules = apriori(dataset, parameter = list(support = 0.004, confidence = 0.2))

#VISUALISING THE RESULT
inspect(sort(rules, by = 'lift')[1:10])