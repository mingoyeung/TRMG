library("qrnn")
library(readxl)
library(ggplot2)
library(openxlsx)

train<- read_excel("C:/Users/ym_ya/Desktop/windpower/Simulation code and data/tfmg/tsub1/train1.xlsx")
val<- read_excel("C:/Users/ym_ya/Desktop/windpower/Simulation code and data/tfmg/tsub1/val1.xlsx")
test<- read_excel("C:/Users/ym_ya/Desktop/windpower/Simulation code and data/tfmg/tsub1/test1.xlsx")
set.seed(12345)
taus <- seq(0.05, 0.95, by=0.05)
train_x <- train[,1:96]
val_x <- val[,1:96]
test_x <- test[,1:96]
train_y <-  train[,97]
val_y <-  val[,97]
test_y <-  test[,97]

X_train<- as.matrix(train_x)
X_val<- as.matrix(val_x)
X_test<- as.matrix(test_x)
Y_train<- as.matrix(train_y)
Y_val<- as.matrix(val_y)
Y_test<- as.matrix(test_y)

fit.mcqrnn <- mcqrnn.fit(X_train, Y_train, tau=taus,n.hidden=10, n.hidden2=5, n.trials=1,iter.max=1000,penalty=2)

#pred.mcqrnn <- mcqrnn.predict(X_val, fit.mcqrnn)
pred.mcqrnn <- mcqrnn.predict(X_test, fit.mcqrnn)
#ytrue_matrix <- matrix(rep(Y_val, each = ncol(pred.mcqrnn)), ncol = ncol(pred.mcqrnn), byrow = TRUE)
ytrue_matrix <- matrix(rep(Y_test, each = ncol(pred.mcqrnn)), ncol = ncol(pred.mcqrnn), byrow = TRUE)
error<-  ytrue_matrix-pred.mcqrnn
ps<-tilted.abs(error, taus)
ps_loss <- mean(ps)
print(ps_loss)


#matplot( cbind(pred.mcqrnn2[,1],pred.mcqrnn2[,19] ,Y_test), type = "l", lty = 1, col = c("red","red", "blue"))
mcqrnnpred <- as.data.frame(pred.mcqrnn)
write.xlsx(mcqrnnpred,file = "mcqrnnpred1.xlsx")