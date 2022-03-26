###### Set working dir and source some functions #######


setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")
source('Experiments/figures/R_scripts/vis_utils.R')


###### Data prep #######


# Import data
benchmark = read_csv('MoleculeACE/Data/results/Benchmark_results.csv')
target_info = read_csv("MoleculeACE/Data/benchmark_data/metadata/Table_1.csv")

# Match target class to dataset 
benchmark$target = target_info$Type[match(benchmark$dataset, gsub('ChEMBL', 'CHEMBL', paste0(target_info$`ChEMBL id`, '_', target_info$Label)))]
benchmark$name = target_info$Name[match(benchmark$dataset, gsub('ChEMBL', 'CHEMBL', paste0(target_info$`ChEMBL id`, '_', target_info$Label)))]

# Rename some stuff
names(benchmark) = gsub('ac-rmse_soft_consensus','cliff_rmse',names(benchmark))
benchmark$descriptor[benchmark$descriptor == 'Canonical'] = 'Graph'
benchmark$descriptor[benchmark$descriptor == 'Attentivefp'] = 'Graph'

benchmark$target = gsub('G_Protein_Coupled_Receptor', 'GPCR', benchmark$target)
benchmark$target = gsub('Nuclear_receptor', 'NR', benchmark$target)
benchmark$target = gsub('Protease', 'Other', benchmark$target)
benchmark$target = gsub('Transferase', 'Other', benchmark$target)
benchmark$target = factor(benchmark$target, levels = c('Kinase', 'GPCR', 'NR', 'Other'))

# Select the DL methods + a good and bad classical method
benchmark_targets = rbind(subset(benchmark, algorithm == 'SVM' & descriptor == 'ECFP'),
                          subset(benchmark, algorithm == 'LSTM' & augmentation == 10),
                          subset(benchmark, algorithm == 'AFP'),
                          subset(benchmark, algorithm == 'MLP'))
benchmark_targets$algorithm = factor(benchmark_targets$algorithm, levels = c('LSTM', 'AFP', 'SVM', 'MLP'))


###### Box plot #######


lstm_target = ggplot(subset(benchmark_targets, algorithm == 'LSTM'), aes(x=target, y=cliff_rmse, fill = target))+
  geom_jitter(aes(color=target), position=position_jitterdodge(0), 
              size=1, shape=1, alpha=0.5) +
  geom_jitter(aes(color=target), position=position_jitterdodge(0), 
              size=1, shape=19, alpha=0.5) +
  geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
               outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=1) +
  scale_y_continuous(breaks = seq(0.5,2,0.25), limits = c(0.5,2), expand = expansion(mult = c(0.01, 0.01)), labels = c('0.5','','1','','1.5','','2')) +
  scale_color_manual(values = rep(descr_cols$cols[which(descr_cols$descr == 'SMILES')], 4))+
  scale_fill_manual(values = rep(descr_cols$cols[which(descr_cols$descr == 'SMILES')], 4))+
  labs(x='Drug target type', y='', fill = 'Target')+
  geom_text(x=1, y=1.8, label="LSTM", color = 'black', size=2.5, fontface="bold", hjust=0) +
  guides(fill = 'none',
         color = 'none')+
  default_theme


gnn_target = ggplot(subset(benchmark_targets, algorithm == 'AFP'), aes(x=target, y=cliff_rmse, fill = target))+
  geom_jitter(aes(color=target), position=position_jitterdodge(0), 
              size=1, shape=1, alpha=0.5) +
  geom_jitter(aes(color=target), position=position_jitterdodge(0), 
              size=1, shape=19, alpha=0.5) +
  geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
               outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=1) +
  scale_y_continuous(breaks = seq(0.5,2,0.25), limits = c(0.5,2), expand = expansion(mult = c(0.01, 0.01)), labels = c('0.5','','1','','1.5','','2')) +
  scale_color_manual(values = rep(descr_cols$cols[which(descr_cols$descr == 'Graph')], 4))+
  scale_fill_manual(values = rep(descr_cols$cols[which(descr_cols$descr == 'Graph')], 4))+
  labs(x='Drug target type', y=bquote("RMSE"[cliff]), fill = 'Target')+
  geom_text(x=1, y=1.8, label="AFP", color = 'black', size=2.5, fontface="bold", hjust=0) +
  guides(fill = 'none',
         color = 'none')+
  default_theme

mlp_target = ggplot(subset(benchmark_targets, algorithm == 'MLP'), aes(x=target, y=cliff_rmse, fill = target))+
  geom_jitter(aes(color=target), position=position_jitterdodge(0), 
              size=1, shape=1, alpha=0.5) +
  geom_jitter(aes(color=target), position=position_jitterdodge(0), 
              size=1, shape=19, alpha=0.5) +
  geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
               outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=1) +
  scale_y_continuous(breaks = seq(0.5,2,0.25), limits = c(0.5,2), expand = expansion(mult = c(0.01, 0.01)), labels = c('0.5','','1','','1.5','','2')) +
  scale_color_manual(values = rep(descr_cols$cols[which(descr_cols$descr == 'ECFP')], 4))+
  scale_fill_manual(values = rep(descr_cols$cols[which(descr_cols$descr == 'ECFP')], 4))+
  labs(x='Drug target type', y='', fill = 'Target')+
  geom_text(x=1, y=1.8, label="DNN", color = 'black', size=2.5, fontface="bold", hjust=0) +
  guides(fill = 'none',
         color = 'none')+
  default_theme


svm_target = ggplot(subset(benchmark_targets, algorithm == 'SVM'), aes(x=target, y=cliff_rmse, fill = target))+
  geom_jitter(aes(color=target), position=position_jitterdodge(0), 
              size=1, shape=1, alpha=0.5) +
  geom_jitter(aes(color=target), position=position_jitterdodge(0), 
              size=1, shape=19, alpha=0.5) +
  geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
               outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=1) +
  scale_y_continuous(breaks = seq(0.5,2,0.25), limits = c(0.5,2), expand = expansion(mult = c(0.01, 0.01)), labels = c('0.5','','1','','1.5','','2')) +
  scale_color_manual(values = rep(descr_cols$cols[which(descr_cols$descr == 'ECFP')], 4))+
  scale_fill_manual(values = rep(descr_cols$cols[which(descr_cols$descr == 'ECFP')], 4))+
  labs(x='Drug target type', y='', fill = 'Target')+
  geom_text(x=1, y=1.8, label="SVM", color = 'black', size=2.5, fontface="bold", hjust=0) +
  guides(fill = 'none',
         color = 'none')+
  default_theme


fig_target = plot_grid(gnn_target, lstm_target, mlp_target, svm_target, ncol=4, nrow=1, scale=1, labels = c('a', 'b', 'c', 'd'), label_size = 10)
print(fig_target)


dev.print(pdf, 'Experiments/figures/sup_fig_6.pdf', width = 7.205, height = 1.752)
