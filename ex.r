#%%
library(arrow, lib.loc = Sys.getenv("R_LIBS_USER"))

data <- read_parquet("/home1/rldnjs16/transit/dataset/data_month/year=2025/month=05/data.parquet")
print(data)

#%%

