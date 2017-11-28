#read datasets
flow<-read.csv("global_flow.csv")
dim(flow)
results<-read.csv("economic factors.csv")
dim(results)

#creating binary variable migrated
migrated<-ifelse(flow$countryflow_1990=="0"&flow$countryflow_1995=="0"&flow$countryflow_2000=="0"&flow$countryflow_2005=="0",0,1)
flow<-data.frame(flow,migrated)

#renaming variables names in results dataset
names(results)[1]<-"Country_name"
names(results)[2]<-"Country_code"
names(results)[3] <- "Avg_mig"
names(results)[4]<-"Health_exp"
names(results)[5]<-"Consum_exp"
names(results)[6]<-"National_incP.C"
names(results)[7]<-"National_inc"
names(results)[8]<-"Prim_enrol_rate"
names(results)[9]<-"Current_accbal"
names(results)[10]<-"Employers_total"
names(results)[11]<-"Employment_serv"
names(results)[12]<-"Employment_indus"
names(results)[13]<-"Employment_agri"
names(results)[14]<-"Expense"
names(results)[15]<-"GDP_growth"
names(results)[16]<-"Hightech_exports"
names(results)[17]<-"HIV_incidence"
names(results)[18]<-"Labor_particip_rate"
names(results)[19]<-"Life_expect_birth"
names(results)[20]<-"Pop_total"
names(results)[21]<-"Pop_density"
names(results)[22]<-"Pop_slums"
names(results)[23]<-"Tax_revenue"
names(results)[24]<-"Tech_grants"
names(results)[25]<-"Lit_rate15_24"
names(results)[26]<-"SnP_equityindex"
names(results)[27]<-"Reg_business"
colnames(results)
results <- results[!apply(is.na(results) | results == "", 1, all),]

#k-means clustering
results.scaled<-scale(data.frame(results$Avg_mig,results$Health_exp,results$Consum_exp,results$National_incP.C,results$National_inc,results$Prim_enrol_rate,
                                 results$Current_accbal,results$Employers_total,results$Employment_serv,results$Employment_indus,results$Employment_agri,
                                 results$Expense,results$GDP_growth,results$Hightech_exports,results$HIV_incidence,results$Labor_particip_rate,results$Life_expect_birth,
                                 results$Pop_total,results$Pop_density,results$Pop_slums,results$Tax_revenue,results$Tech_grants,results$Lit_rate15_24,results$SnP_equityindex,
                                 results$Reg_business))
colnames(results.scaled)
totwss<-vector()
btwss<-vector()
for(i in 3:27)
{
  set.seed(1234)
  temp<-kmeans(results.scaled,centers=i)
  totwss[i]<-temp$tot.withinss
  btwss[i]<-temp$betweenss
}
plot(totwss,xlab="number of cluster",type="b",ylab="total within sum of square")
plot(btwss,xlab="number of cluster",type="b",ylab="total between sum of square")

#installing rserve to connect with tableau
#install.packages("Rserve")
library(Rserve)
Rserve()

#Decision tree
library(rpart)
library(rattle)
fit <- rpart(migration ~ Avg_mig+Health_exp+Consum_exp+National_incP.C+National_inc+Prim_enrol_rate+Current_accbal+Employers_total+Employment_serv
             +Employment_indus+Employment_agri+Expense+GDP_growth+Hightech_exports+HIV_incidence+Labor_particip_rate+Life_expect_birth+Pop_total
             +Pop_density+Pop_slums+Tax_revenue+Tech_grants+Lit_rate15_24+SnP_equityindex+Reg_business,method="class",
             data=results)
printcp(fit)
plotcp(fit)
summary(fit)
decision_tree<-results
set.seed(1)
test=sample(1:nrow(decision_tree),nrow(decision_tree)/4)
train=-test
training_data=decision_tree[train,]
testing_data=decision_tree[test,]
testing_survived=decision_tree$mig_class[test]
tree_predict=predict(fit,testing_data,type="class")
mean(tree_predict!=testing_survived)

#visualizing the Decision tree
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(fit,sub="decision_tree")
plot(fit, uniform=TRUE, 
     main="Classification Tree for Factors")
text(fit, use.n=TRUE, all=TRUE, cex=.8)
pfit<- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
plot(pfit, uniform=TRUE, 
     main="Pruned Classification Tree")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)




