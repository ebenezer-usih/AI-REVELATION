#ECLAT

#DATA PREPROCESSING
install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
itemFrequencyPlot(dataset, topN = 10)

#TRAINING ECLAT ON THE DATASET
rules = eclat(dataset, parameter = list(support = 0.003, minlen = 2))

#VISUALISING THE RESULT
inspect(sort(rules, by = 'support')[1:10])