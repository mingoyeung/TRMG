library("qrnn")
library(readxl)
library(ggplot2)
library(openxlsx)

train<- read_excel("C:/Users/ym_ya/Desktop/windpower/Simulation code and data/tvfemd_R2CMSE/tsub1/train1.xlsx")
val<- read_excel("C:/Users/ym_ya/Desktop/windpower/Simulation code and data/tvfemd_R2CMSE/tsub1/val1.xlsx")
test<- read_excel("C:/Users/ym_ya/Desktop/windpower/Simulation code and data/tvfemd_R2CMSE/tsub1/test1.xlsx")
set.seed(12345)

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

tau <- seq(0.05, 0.95, by=0.05)
w <- pred.qrnn <- vector("list", length(tau))

for(i in seq_along(tau)){
  w[[i]] <- qrnn.fit(x=X_train, y=Y_train, n.hidden=1,
                      tau=tau[i], iter.max=1000, n.trials=1, penalty=1)
}

results <- vector("list", length(tau))
for(i in seq_along(tau)){
  pred.qrnn[[i]] <- qrnn.predict(X_test, w[[i]])#val
}

results <- do.call(cbind, pred.qrnn)
#ytrue_matrix <- matrix(rep(Y_val, each = ncol(results)), ncol = ncol(results), byrow = TRUE)
ytrue_matrix <- matrix(rep(Y_test, each = ncol(results)), ncol = ncol(results), byrow = TRUE)
error<-  ytrue_matrix-results
ps<-tilted.abs(error, tau)
ps_loss <- mean(ps)
print(ps_loss)

#matplot( cbind(qr2nn[,1],qr2nn[,19] ,Y_test), type = "l", lty = 1, col = c("red","red", "blue"))
qrnnpred <- as.data.frame(results)
write.xlsx(qrnnpred,file = "trqg1.xlsx")