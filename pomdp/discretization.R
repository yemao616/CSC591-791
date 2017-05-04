
library(infotheo)
s.data <- read.csv( '/Users/Ye/Dropbox/CSC791_projects/Assigned project/FAMD/reduce.famd.features.data.csv')
k.data <- s.data
# Get discretized data
for(i in 7:56)
  s.data[,i] <- discretize(s.data[,i],"equalfreq",3)

for(i in 7:56)
  k.data[,i] <- kmeans(k.data[,i],3)$cluster
 
prefix <- '/Users/Ye/Dropbox/CSC791_projects/Assigned project/FAMD/'
filename <- paste(prefix,'discretize.csv' ,sep="")
write.csv(s.data, filename, row.names = FALSE, col.names = TRUE, sep = ",")

filename <- paste(prefix,'kmeans.csv' ,sep="")
write.csv(k.data, filename, row.names = FALSE, col.names = TRUE, sep = ",")