
###### Set working dir and source some functions #######


setwd("/Users/derek/Dropbox/PycharmProjects/Activity_cliffs/")
# setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")

source('Experiments/figures/R_scripts/vis_utils.R')

# Import data
benchmark = read_csv('MoleculeACE/Data/results/Benchmark_results.csv')
names(benchmark) = gsub('ac-rmse_soft_consensus','cliff_rmse',names(benchmark))
benchmark$descriptor[benchmark$descriptor == 'Canonical'] = 'Graph'
benchmark$descriptor[benchmark$descriptor == 'Attentivefp'] = 'Graph'


# CHEMBL4005_Ki
benchmark$descriptor = factor(benchmark$descriptor, levels = c("Graph","ECFP","MACCs","Physchem","SMILES","WHIM" ))
colours = descr_cols$cols[match(levels(benchmark$descriptor), descr_cols$descr)]

# Plot RMSE RMSEcliff scatters
p_good = ggplot(subset(benchmark, dataset == "CHEMBL239_EC50"), aes(x = rmse, y = cliff_rmse))+
  geom_point(size=1, shape=1, alpha=0.5, aes(color = descriptor))+
  geom_point(size=1, shape=19, alpha=0.5, aes(color = descriptor))+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  labs(x='', y=bquote("RMSE"[cliff]), fill = 'Descriptor')+
  scale_color_manual(values = colours)+
  scale_fill_manual(values = colours)+
  guides(fill = 'none', color = 'none')+
  scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  default_theme

p_bad = ggplot(subset(benchmark, dataset == "CHEMBL2971_Ki"), aes(x = rmse, y = cliff_rmse))+
  geom_point(size=1, shape=1, alpha=0.5, aes(color = descriptor))+
  geom_point(size=1, shape=19, alpha=0.5, aes(color = descriptor))+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  labs(x='RMSE', y=bquote("RMSE"[cliff]), fill = 'Descriptor')+
  scale_color_manual(values = colours)+
  scale_fill_manual(values = colours)+
  guides(fill = 'none', color = 'none')+
  scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  default_theme

benchmark_delta = benchmark

# Match dataset abbreviation to its name
dataset_abbrv = data.frame(abbrv = c('AR','CB1','FX','DOR','D3R','D4R','DAT','CLK4','FXR','GHSR','GR','GSK3','HRH1',
                                     'HRH3','JAK1','JAK2', 'KOR (a)','KOR (i)','MOR','OX2R','PPARa','PPARy','PPARd'
                                     ,'PIK3CA','PIM1','5-HT1A','SERT','SOR','Thrombin','ABL1'),
                           dataset = c('CHEMBL1871_Ki','CHEMBL218_EC50','CHEMBL244_Ki','CHEMBL236_Ki','CHEMBL234_Ki',
                                       'CHEMBL219_Ki','CHEMBL238_Ki', 'CHEMBL4203_Ki','CHEMBL2047_EC50',
                                       'CHEMBL4616_EC50','CHEMBL2034_Ki','CHEMBL262_Ki','CHEMBL231_Ki','CHEMBL264_Ki',
                                       'CHEMBL2835_Ki','CHEMBL2971_Ki','CHEMBL237_EC50','CHEMBL237_Ki','CHEMBL233_Ki',
                                       'CHEMBL4792_Ki','CHEMBL239_EC50', 'CHEMBL3979_EC50','CHEMBL235_EC50',
                                       'CHEMBL4005_Ki','CHEMBL2147_Ki','CHEMBL214_Ki','CHEMBL228_Ki','CHEMBL287_Ki',
                                       'CHEMBL204_Ki','CHEMBL1862_Ki'))
benchmark_delta$label = dataset_abbrv$abbrv[match(benchmark_delta$dataset, dataset_abbrv$dataset)]

# Calculate RMSE delta
benchmark$rmse_delta = benchmark$cliff_rmse - benchmark$rmse

# Order datasets
mean_dataset <- benchmark_delta %>% group_by(label) %>%  summarise(MinDataset = cor(cliff_rmse, rmse))
# mean_dataset <- benchmark_delta %>% group_by(label) %>%  summarise(MinDataset = median(rmse_delta, na.rm = T))
dataset_order = mean_dataset$label[order(-mean_dataset$MinDataset)]
benchmark_delta$label = factor(benchmark_delta$label, levels = dataset_order)

benchmark_delta$corr = mean_dataset$MinDataset[match(benchmark_delta$label, mean_dataset$label)]
benchmark_delta$corr = as.character(round(benchmark_delta$corr,2))
benchmark_delta$corr = gsub('0\\.', '\\.', benchmark_delta$corr)

benchmark_delta$descriptor = factor(benchmark_delta$descriptor, levels = c("Graph","ECFP","MACCs","Physchem","SMILES","WHIM" ))
colours = descr_cols$cols[match(levels(benchmark_delta$descriptor), descr_cols$descr)]

# Make plot
diff_per_ds = ggplot(benchmark_delta, aes(x=rmse_delta, y=label, fill = descriptor))+

  geom_point(aes(color=descriptor), size=1, shape=1, alpha=0.5) +
  geom_point(aes(color=descriptor), size=1, shape=19, alpha=0.5) +
  
  geom_text_repel(aes(x = -0.5, label = corr), 
                  size = 1.5, 
                  segment.size = 0.25, force = 0, fontface="bold", max.iter = 10, 
                  max.overlaps = 30, show.legend = FALSE)+
  
  geom_vline(xintercept = -0.25, linetype='dashed', alpha=0.25)+
  geom_vline(xintercept = 0, linetype='dashed', alpha=0.75)+
  geom_vline(xintercept = 0.25, linetype='dashed', alpha=0.25) +
  scale_y_discrete(position = "left") +
  scale_x_continuous(labels = c('-0.5', '', '0', '', '0.5')) +
  scale_color_manual(values = colours)+
  scale_fill_manual(values = colours)+
  labs(y='Dataset', x=bquote("RMSE"[cliff]~"- RMSE"), fill = 'Descriptor')+
  guides(fill = 'none', color = 'none')+
  coord_cartesian(xlim=c(-0.5, 0.5))+
  default_theme +
  theme(axis.text.y = element_text(size=6, face="plain", colour = "black"),
        axis.title.y = element_text(size=0, face="plain", colour = "black"))


scatters = plot_grid(p_good, p_bad,
                     label_size = 10, labels = c('b', 'c'),
                     ncol=1, nrow =2, scale = 1)

scatters = plot_grid(diff_per_ds, scatters,
                     label_size = 10, labels = c('a', ''),
                     ncol=2, nrow =1, scale = 1)

print(scatters)

dev.print(pdf, 'Experiments/figures/Fig_6.pdf', width = 3.504, height = 3.504)


for (i in unique(benchmark$dataset)){
  print(i)
  print(cor(subset(benchmark, dataset == i)$rmse, subset(benchmark, dataset == i)$cliff_rmse  ))
}

# 
# # unique(benchmark$dataset)[19]
# 
# subset(benchmark, dataset == unique(benchmark$dataset)[27])$n_soft_consensus_cliff_compounds_test
# subset(benchmark, dataset == unique(benchmark$dataset)[27])$n_compounds_test
# 
# 
# benchmark_no_pretrain <- read_csv("Dropbox/PycharmProjects/Activity_cliffs/Results/Benchmark_results_no_pretrain.csv")
# 
# pt = subset(benchmark_10_aug, algorithm == 'lstm' & augmentation == 10)
# pt$pretrain = 'Yes'
# npt = subset(benchmark_no_pretrain, algorithm == 'lstm' & augmentation == 10)
# npt$pretrain = 'No'
# 
# benchmark_pre_training = rbind(pt, npt)
# names(benchmark_pre_training) = gsub('ac-rmse_soft_consensus','cliff_rmse',names(benchmark_pre_training))
# 
# lstm_pretrain = ggplot(benchmark_pre_training, aes(y = cliff_rmse, x = pretrain, group=dataset))+
#     geom_point(size = 2, shape=1, alpha=0.5, color='#3bb1e6') +
#     geom_point(size = 2, shape=19, alpha=0.5, color='#3bb1e6') +
#     geom_line(alpha=0.2, size = 0.75, color='#3bb1e6') +
#     geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.15,
#                  outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=1, aes(group=pretrain), fill='#3bb1e6') +
#     labs(x='Pre-training', y='') +
#     scale_y_continuous(breaks = seq(0,5,0.5), expand = expansion(mult = c(0.01, 0.01))) +
#     coord_cartesian(ylim=c(0, 2))+
#     default_theme