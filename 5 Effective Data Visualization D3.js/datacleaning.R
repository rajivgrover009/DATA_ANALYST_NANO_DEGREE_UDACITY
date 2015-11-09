# Load the library for data cleaning
library('tidyr')

# Read csv fil
co2<-read.csv(file=file.path("E:","DataScienceWithR/Nano Degree Udacity/Projects
                             /Project 6/g7_co2_emissions_.csv"),header=TRUE,
                                stringsAsFactors=FALSE)
#check the structure of hte file.
str(co2)

# construct hte column names to be selected from the file
myvars <- c("countryName","X1961","X1965","X1970","X1975","X1980","X1985",
            "X1990","X1995","X2000","X2005","X2010")

#filter the dataframe
df<-co2[myvars]

#Reset the column names
names(df)<-c("countryName","1961","1965","1970","1975","1980","1985","1990",
             "1995","2000","2005","2010")

#transpose the data
df<-tidyr::gather(df,"Year","co2",2:12)

# order the data
df<-df[order(df$countryName),]

#Export in to csv file.
write.csv(df, file = "MyData_updated.csv")

