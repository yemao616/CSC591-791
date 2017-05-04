prefix <- '/Users/Ye/🍃ymao4/📚Research/pomdp_code/'
filename <- paste(prefix,'wis.csv', sep="")
data <- read.csv(filename)

boxplot(data, las = 2,main="WIS over different number of features", 
        ylab="Weighted IS-estimated reward")
