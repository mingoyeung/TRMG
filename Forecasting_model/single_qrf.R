library(quantregForest)
library(readxl)
library(ggplot2)
library(openxlsx)
start_time<-Sys.time()
train<- read_excel("C:/Users/ym_ya/Desktop/windpower/Simulation code and data/single_model/train.xlsx")
val<- read_excel("C:/Users/ym_ya/Desktop/windpower/Simulation code and data/single_model/val.xlsx")
test<- read_excel("C:/Users/ym_ya/Desktop/windpower/Simulation code and data/single_model/test.xlsx")
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

tau_values <-seq(0.05, 0.95, by = 0.05)
qrf <- quantregForest(x=X_train, y=Y_train, nodesize=20)

end_time<-Sys.time()
total_time<-end_time-start_time
total_time
conditionalQuantiles <- predict(qrf, X_val, what=tau_values)
inference_start <- Sys.time()
conditionalQuantiles2 <- predict(qrf, X_test, what=tau_values)
inference_end <- Sys.time()
total_inference<-inference_end -inference_start
total_inference
#matplot( cbind(conditionalQuantiles2[,1],conditionalQuantiles2[,19] ,Y_test), type = "l", lty = 1, col = c("red","red", "blue"))
#mcqrnnpred <- as.data.frame(conditionalQuantiles2)
#write.xlsx(mcqrnnpred,file = "C:/Users/ym_ya/Desktop/code and data/single_model/qrfpred.xlsx")