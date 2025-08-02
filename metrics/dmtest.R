library(qrnn)
library(openxlsx)
library(readxl)
library(forecast)

trmg <- read_excel("C:/Users/ym_ya/Desktop/Simulation code and data/tvfemd_R2CMSE/TVFEMD_R2CMSE_MCQRNN_ensemble_result.xlsx")
trqg <- read_excel("C:/Users/ym_ya/Desktop/Simulation code and data/tvfemd_R2CMSE/TVFEMD_R2CMSE_QRNN2_ensemble_result.xlsx")
tsmg <- read_excel("C:/Users/ym_ya/Desktop/Simulation code and data/TVFEMD_SE_MCQRNNG(TSMG)/tvfemd_se_mcqrnng.xlsx")
ermg <- read_excel("C:/Users/ym_ya/Desktop/Simulation code and data/EMD_R2CMSE_MCQRNNG(ERMG)/EMD_R2CMSE_MCQRNN_ensemble_result.xlsx")
mcqr <- read_excel("C:/Users/ym_ya/Desktop/Simulation code and data/single_model/mcqrnnpred.xlsx")
qr2nn <- read_excel("C:/Users/ym_ya/Desktop/Simulation code and data/single_model/qrnnpred.xlsx")
qrf <- read_excel("C:/Users/ym_ya/Desktop/Simulation code and data/single_model/qrfpred.xlsx")
qrlasso <- read_excel("C:/Users/ym_ya/Desktop/Simulation code and data/single_model/qrlassopred.xlsx")
actual <- read_excel("C:/Users/ym_ya/Desktop/Simulation code and data/actual.xlsx")

model1<- read_excel("C:/Users/ym_ya/Desktop/code and data/dmtest/ploss1.xlsx")
model2<- read_excel("C:/Users/ym_ya/Desktop/code and data/dmtest/ploss8.xlsx")
dm_results <- numeric(19)

for (i in 1:19) {
  
  model1_loss <- as.matrix(model1[, i])
  model2_loss <- as.matrix(model2[, i])
  
  dm_test_result <- dm.test(model1_loss, model2_loss,alternative = "two.sided",h=1,power=1,varestimator="acf")
  
  dm_results[i] <- dm_test_result$statistic
}

print(dm_results)
taus <- seq(0.05, 0.95, by = 0.05)

actual<- as.matrix(actual)

trmg<- as.matrix(trmg)
ytrue <- matrix(rep(actual, each = ncol(trmg)), ncol = ncol(trmg), byrow = TRUE)
error1 <- trmg - ytrue
ps1 <- tilted.abs(error1, taus)

trqg<- as.matrix(trqg)
error2 <- trqg - ytrue
ps2 <- tilted.abs(error2, taus)

tsmg<- as.matrix(tsmg)
error3 <- tsmg - ytrue
ps3 <- tilted.abs(error3, taus)

ermg<- as.matrix(ermg)
error4 <- ermg - ytrue
ps4 <- tilted.abs(error4, taus)

mcqr<- as.matrix(mcqr)
error5 <- mcqr - ytrue
ps5 <- tilted.abs(error5, taus)

qr2nn<- as.matrix(qr2nn)
error6 <- qr2nn - ytrue
ps6 <- tilted.abs(error6, taus)

qrlasso<- as.matrix(qrlasso)
error7 <- qrlasso - ytrue
ps7 <- tilted.abs(error7, taus)

qrf<- as.matrix(qrf)
error8 <- qrf - ytrue
ps8 <- tilted.abs(error8, taus)

ploss1 <- as.data.frame(ps1)
ploss2 <- as.data.frame(ps2)
ploss3 <- as.data.frame(ps3)
ploss4 <- as.data.frame(ps4)
ploss5 <- as.data.frame(ps5)
ploss6 <- as.data.frame(ps6)
ploss7 <- as.data.frame(ps7)
ploss8 <- as.data.frame(ps8)

write.xlsx(ploss1,file = "C:/Users/ym_ya/Desktop/code and data/dmtest/ploss1.xlsx")
write.xlsx(ploss2,file = "C:/Users/ym_ya/Desktop/code and data/dmtest/ploss2.xlsx")
write.xlsx(ploss3,file = "C:/Users/ym_ya/Desktop/code and data/dmtest/ploss3.xlsx")
write.xlsx(ploss4,file = "C:/Users/ym_ya/Desktop/code and data/dmtest/ploss4.xlsx")
write.xlsx(ploss5,file = "C:/Users/ym_ya/Desktop/code and data/dmtest/ploss5.xlsx")
write.xlsx(ploss6,file = "C:/Users/ym_ya/Desktop/code and data/dmtest/ploss6.xlsx")
write.xlsx(ploss7,file = "C:/Users/ym_ya/Desktop/code and data/dmtest/ploss7.xlsx")
write.xlsx(ploss8,file = "C:/Users/ym_ya/Desktop/code and data/dmtest/ploss8.xlsx")