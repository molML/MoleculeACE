###### Set working dir and source some functions #######


setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")
source('Experiments/figures/R_scripts/vis_utils.R')


###### Data prep #######


# Import data
benchmark = read_csv('MoleculeACE/Data/results/Benchmark_results.csv')

# Rename some stuff
names(benchmark) = gsub('ac-rmse_soft_consensus','cliff_rmse',names(benchmark))

# Select only the classical methods
benchmark_ml = subset(benchmark, algorithm %in% c('RF', 'SVM', 'GBM', 'KNN'))

# order factors of algorithm and descriptor
mean_algo = benchmark_ml %>% group_by(algorithm) %>% summarise_all("mean")
mean_descr = benchmark_ml %>% group_by(descriptor) %>% summarise_all("mean")

algo_order = mean_algo[rev(order(mean_algo$cliff_rmse)),]$algorithm
descr_order = mean_descr[rev(order(mean_descr$cliff_rmse)),]$descriptor

# re-oder factors based on mean performance
benchmark_ml$algorithm = factor(benchmark_ml$algorithm, levels = algo_order)
benchmark_ml$descriptor = factor(benchmark_ml$descriptor, levels = descr_order)


###### Boxplot #######


colours = descr_cols$cols[match(c('WHIM', 'Physchem', 'MACCs', 'ECFP'), descr_cols$descr)]

box_plot_ml = ggplot(benchmark_ml, aes(x=algorithm, y=rmse, fill = descriptor))+
  
  geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), 
              size=1, shape=1, alpha=0.5) +
  geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), 
              size=1, shape=19, alpha=0.5) +
  
  geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.5,
               outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=0.75) +
  
  scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75),
                     expand = expansion(mult = c(0.01, 0.01)))+

  scale_color_manual(values = colours)+
  scale_fill_manual(values = colours)+
  labs(x='Algorithm', y=bquote("RMSE"), fill = 'Descriptor')+
  guides(fill = 'none', color = 'none')+
  default_theme

print(box_plot_ml)

# outlier is CHEMBL2835

##### DL ######

# Select only the classical methods
benchmark_dl = rbind(subset(benchmark, algorithm %in% c('GCN', 'MPNN', 'AFP', "GAT", 'MLP')),
                     subset(benchmark, algorithm == 'SVM' & descriptor == 'ECFP'),
                     subset(benchmark, algorithm == 'LSTM' & augmentation == 10),
                     subset(benchmark, algorithm == 'CNN' & augmentation == 10))

benchmark_dl$descriptor[benchmark_dl$descriptor == 'Canonical'] = 'Graph'
benchmark_dl$descriptor[benchmark_dl$descriptor == 'Attentivefp'] = 'Graph'

# order factors of algorithm and descriptor
algo_order = c('GAT', 'GCN', 'MPNN', 'AFP', 'CNN', 'LSTM', 'MLP', "SVM")
benchmark_dl$algorithm = factor(benchmark_dl$algorithm, levels = algo_order)
benchmark_dl$descriptor = factor(benchmark_dl$descriptor, levels = c('ECFP', 'SMILES', 'Graph'))

colours = descr_cols$cols[match(c('ECFP', 'SMILES', 'Graph'), descr_cols$descr)]

# The plot
box_plot_dl = ggplot(benchmark_dl, aes(x=algorithm, y=rmse, fill=descriptor))+
  geom_jitter(aes(color=descriptor), position=position_jitterdodge(0),
              size=1, shape=1, alpha=0.5) +
  geom_jitter(aes(color=descriptor), position=position_jitterdodge(0),
              size=1, shape=19, alpha=0.5) +
  geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
               outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=1) +
  
  scale_y_continuous(breaks = seq(0,3,0.25), 
                     expand = expansion(mult = c(0.01, 0.01)),
                     labels = c('0.00', '', '0.50', '', '1.00', '', '1.50', '', '2.00', '', '2.50', '', '3.00'))+
  coord_cartesian(ylim=c(0.5, 2.5))+
  
  scale_color_manual(values = colours)+
  scale_fill_manual(values = colours)+
  labs(x='Algorithm', y=bquote("RMSE"))+
  guides(fill = 'none', 
         color = 'none')+
  default_theme

print(box_plot_dl)

boxplots = plot_grid(box_plot_ml, box_plot_dl, ncol=2, nrow=1, scale=1, labels = c('a', 'b'), label_size = 10)
print(boxplots)

dev.print(pdf, 'Experiments/figures/sup_fig_1', width = 7.205, height = 2.5)







