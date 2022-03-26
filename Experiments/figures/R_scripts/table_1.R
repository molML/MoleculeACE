library(readr)
library(plyr)
library(dplyr)

###### Set working dir and source some functions #######


setwd("/Users/derek/Dropbox/PycharmProjects/Activity_cliffs/")
# setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")

benchmark <- read_csv("Results/Benchmark_results.csv")
target_info = read_csv("Data/metadata/dataset_info.csv")

benchmark = benchmark %>% group_by(dataset) %>% summarise_all("mean")

benchmark$target = target_info$Class[match(benchmark$dataset, target_info$Dataset)]
benchmark$name = target_info$Name[match(benchmark$dataset, target_info$Dataset)]

benchmark$target = gsub('G_Protein_Coupled_Receptor', 'GPCR', benchmark$target)
benchmark$target = gsub('Nuclear_receptor', 'NR', benchmark$target)

# Name Class label CHEMBL_id cpds_train cpds_test cliff_cpds_train cliff_cpds_test
new_df = benchmark[c('name', 'target', 'dataset', 'dataset', "n_compounds_train", "n_compounds_test", "n_soft_consensus_cliff_compounds_train", "n_soft_consensus_cliff_compounds_test")]
names(new_df) = c('Name', 'Type', 'Label', 'ChEMBL id', 'Train compounds', 'Test compounds', 'Train activity cliff compounds', 'Test activity cliff compounds')

new_df$Label = gsub(".*_", "", new_df$Label)
new_df$`ChEMBL id` = gsub("_.*", "", new_df$`ChEMBL id`)
new_df$`ChEMBL id` = gsub("H", "h", new_df$`ChEMBL id`)

new_df = new_df[order(new_df$`Train compounds`, decreasing = T),]

write_csv(new_df, 'Experiments/figures/Table_1.csv')
