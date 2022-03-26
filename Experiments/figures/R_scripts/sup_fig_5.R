###### Set working dir and source some functions #######


setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")
source('Experiments/figures/R_scripts/vis_utils.R')


###### Data prep #######


# Import data
benchmark = read_csv('MoleculeACE/Data/results/Benchmark_results.csv')
target_info = read_csv("Experiments/figures/Table_1.csv")


# Match target class to dataset 
benchmark$target = target_info$Type[match(benchmark$dataset, gsub('ChEMBL', 'CHEMBL', paste0(target_info$`ChEMBL id`, '_', target_info$Label)))]
benchmark$name = target_info$Name[match(benchmark$dataset, gsub('ChEMBL', 'CHEMBL', paste0(target_info$`ChEMBL id`, '_', target_info$Label)))]


# Rename some stuff
names(benchmark) = gsub('ac-rmse_soft_consensus','cliff_rmse',names(benchmark))
benchmark$descriptor[benchmark$descriptor == 'Canonical'] = 'Graph'
benchmark$descriptor[benchmark$descriptor == 'Attentivefp'] = 'Graph'


##### Dataset size ######

gnn_scatter = ggplot(subset(benchmark, algorithm == 'AFP'), aes(x = n_compounds_train, y = cliff_rmse))+
  geom_point(size=1, shape=1, alpha=0.5, color=descr_cols$cols[which(descr_cols$descr == 'Graph')]) +
  geom_point(size=1, shape=19, alpha=0.5, color=descr_cols$cols[which(descr_cols$descr == 'Graph')]) +
  labs(x='', y=bquote("RMSE"[cliff]))+
  geom_text(x=200, y=1.8, label="AFP", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_y_continuous(breaks = seq(0,2,0.5), limits = c(0.01,2), expand = expansion(mult = c(0.01, 0.01))) +
  scale_x_continuous(breaks = seq(0,3000, 1000), limits = c(0.01,3000), 
                     expand = expansion(mult = c(0.01, 0.01)))+
  default_theme

lstm_scatter = ggplot(subset(benchmark, algorithm == 'LSTM'), aes(x = n_compounds_train, y = cliff_rmse))+
  geom_point(size=1, shape=1, alpha=0.5, color=descr_cols$cols[which(descr_cols$descr == 'SMILES')]) +
  geom_point(size=1, shape=19, alpha=0.5, color=descr_cols$cols[which(descr_cols$descr == 'SMILES')]) +
  labs(x='', y='')+
  geom_text(x=200, y=1.8, label="LSTM", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_y_continuous(breaks = seq(0,2,0.5), limits = c(0.01,2), expand = expansion(mult = c(0.01, 0.01))) +
  scale_x_continuous(breaks = seq(0,3000, 1000), limits = c(0.01,3000), 
                     expand = expansion(mult = c(0.01, 0.01)))+
  default_theme

mlp_scatter = ggplot(subset(benchmark, algorithm == 'MLP'), aes(x = n_compounds_train, y = cliff_rmse))+
  geom_point(size=1, shape=1, alpha=0.5, color=descr_cols$cols[which(descr_cols$descr == 'ECFP')]) +
  geom_point(size=1, shape=19, alpha=0.5, color=descr_cols$cols[which(descr_cols$descr == 'ECFP')]) +
  labs(x='Train compounds', y=bquote("RMSE"[cliff]))+
  geom_text(x=200, y=1.8, label="MLP", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_y_continuous(breaks = seq(0,2,0.5), limits = c(0.01,2), expand = expansion(mult = c(0.01, 0.01))) +
  scale_x_continuous(breaks = seq(0,3000, 1000), limits = c(0.01,3000), 
                     expand = expansion(mult = c(0.01, 0.01)))+
  default_theme

svm_scatter = ggplot(subset(benchmark, algorithm == 'SVM' & descriptor == 'ECFP'), aes(x = n_compounds_train, y = cliff_rmse))+
  geom_point(size=1, shape=1, alpha=0.5, color=descr_cols$cols[which(descr_cols$descr == 'ECFP')]) +
  geom_point(size=1, shape=19, alpha=0.5, color=descr_cols$cols[which(descr_cols$descr == 'ECFP')]) +
  labs(x='Train compounds', y='')+
  geom_text(x=200, y=1.8, label="SVM", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_y_continuous(breaks = seq(0,2,0.5), limits = c(0.01,2), expand = expansion(mult = c(0.01, 0.01))) +
  scale_x_continuous(breaks = seq(0,3000, 1000), limits = c(0.01,3000), 
                     expand = expansion(mult = c(0.01, 0.01)))+
  default_theme


fig = plot_grid(gnn_scatter, lstm_scatter, mlp_scatter, svm_scatter, ncol=2, nrow=2, scale=1, labels = c('a', 'b', 'c', 'd'), label_size = 10)
print(fig)

dev.print(pdf, 'Experiments/figures/sup_fig_5.pdf', width = 3.504, height = 3.504)
