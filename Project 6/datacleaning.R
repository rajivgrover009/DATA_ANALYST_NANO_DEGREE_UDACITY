
library('tidyr')
co2<-read.csv(file=file.path("E:","DataScienceWithR/Nano Degree Udacity/Projects/Project 6/g7_co2_emissions_.csv"),header=TRUE,stringsAsFactors=FALSE)
str(co2)
# myvars <- c("countryName","X1961","X1970","X1980","X1990","X2000","X2010")
myvars <- c("countryName","X1961","X1965","X1970","X1975","X1980","X1985","X1990","X1995","X2000","X2005","X2010")


df<-co2[myvars]
#names(df)<-c("countryName","1961","1970","1980","1990","2000","2010")
names(df)<-c("countryName","1961","1965","1970","1975","1980","1985","1990","1995","2000","2005","2010")


df<-tidyr::gather(df,"Year","co2",2:12)
df<-df[order(df$countryName),]
write.csv(df, file = "MyData_updated.csv")
df
