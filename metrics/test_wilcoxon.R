library(readxl)
library(tidyverse)
library(broom)

df1 <- read_excel("C:/Users/ym_ya/Desktop/pair.xlsx",sheet="rmse")
df2 <- read_excel("C:/Users/ym_ya/Desktop/pair.xlsx",sheet="mae")
df3 <- read_excel("C:/Users/ym_ya/Desktop/pair.xlsx",sheet="crps")
set.seed(12345)
shapiro.test(df1$trmg-df1$trqg)
shapiro.test(df2$trmg-df2$trqg)
shapiro.test(df3$trmg-df2$trqg)


file_path <- "C:/Users/ym_ya/Desktop/pair.xlsx"
sheet_names <- excel_sheets(file_path)

all_results <- list()

for (sheet in sheet_names) {

  df <- read_excel(file_path, sheet = sheet, skip = 1) %>%
    set_names(c("lasso", "qrf", "qr2nn", "mcqrnn", "ermg", "tsmg", "trqg", "trmg"))
  
  sheet_results <- tibble(
    sheet_name = character(),
    column = character(),
    p_value = numeric(),
    statistic = numeric()
  )
  
  trmg_data <- df$trmg
  
  for (col_name in setdiff(names(df), "trmg")) {

    col_data <- df[[col_name]]
    
    test_result <- wilcox.test(
      x = col_data,
      y = trmg_data,
      paired = TRUE,
      exact = FALSE,
      conf.int = FALSE
    )
    

    sheet_results <- sheet_results %>%
      add_row(
        sheet_name = sheet,
        column = col_name,
        p_value = test_result$p.value,
        statistic = test_result$statistic
      )
  }
  
  all_results[[sheet]] <- sheet_results
}

final_results <- bind_rows(all_results) %>%
  mutate(
    significance = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

print(final_results, n = Inf)