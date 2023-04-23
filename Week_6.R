setwd()
library(tidyverse)
dt = read.csv("C:\\Users\\91799\\Downloads\\education_datascience.csv")
str(dt)
df <- dt [dt$ 'Educational Attainment', 'Children under 15']
dt %>% drop_na("Population.Count")
barplot(names.arg = dt$Gender, dt$Population.Count)

gender.count<-table(dt$Gender)
barplot(gender.count, names.arg=c('Male','Female'), main='Distribution of gender', xlab='Gender', ylab='Count')
