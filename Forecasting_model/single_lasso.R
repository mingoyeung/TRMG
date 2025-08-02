library(quantreg)
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

tau_values <- seq(0.05, 0.95, by=0.05)
results <- list() 

for (tau in tau_values) {
  result <- rq.fit.lasso(X_train, Y_train, tau = tau, lambda = 2, beta = .99995, eps = 1e-06)
  results[[paste0("tau_", tau)]] <- result  
}
end_time<-Sys.time()
total_time<-end_time-start_time
total_time
predictions <- list()
inference_start <- Sys.time()
for (tau in tau_values) {
  coefficients <- results[[paste0("tau_", tau)]]$coefficients

  prediction <- as.vector(X_test %*% coefficients)#X_test
  
  predictions[[paste0("tau_", tau)]] <- prediction
}
inference_end <- Sys.time()
total_inference<-inference_end -inference_start
total_inference
predictions_matrix <- cbind(
  predictions$tau_0.05,
  predictions$tau_0.1,
  predictions$tau_0.15,  
  predictions$tau_0.2,
  predictions$tau_0.25,
  predictions$tau_0.3,  
  predictions$tau_0.35,
  predictions$tau_0.4,
  predictions$tau_0.45,  
  predictions$tau_0.5,
  predictions$tau_0.55,
  predictions$tau_0.6,  
  predictions$tau_0.65,
  predictions$tau_0.7,
  predictions$tau_0.75,  
  predictions$tau_0.8,
  predictions$tau_0.85,
  predictions$tau_0.9,
  predictions$tau_0.95
)

colnames(predictions_matrix) <- c("tau=0.05", "tau=0.1", "tau=0.15","tau=0.2", 
                                  "tau=0.25", "tau=0.3","tau=0.35", "tau=0.4", 
                                  "tau=0.45","tau=0.5", "tau=0.55", "tau=0.6",
                                  "tau=0.65", "tau=0.7", "tau=0.75","tau=0.8", 
                                  "tau=0.85", "tau=0.9","tau=0.95")

#print(predictions_matrix)
#matplot( cbind(predictions_matrix2[,1],predictions_matrix2[,19] ,Y_test), type = "l", lty = 1, col = c("red","red", "blue"))
#mcqrnnpred <- as.data.frame(predictions_matrix2)
#write.xlsx(mcqrnnpred,file = "C:/Users/ym_ya/Desktop/code and data/single_model/qrlassopred.xlsx")
