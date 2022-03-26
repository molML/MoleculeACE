###### Set working dir and source some functions #######

setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")
source('Experiments/figures/R_scripts/vis_utils.R')

###### Data prep #######


# Import data
benchmark = read_csv('MoleculeACE/Data/results/Benchmark_results.csv')

# Rename some stuff
benchmark$descriptor[benchmark$descriptor == 'Canonical'] = 'Graph'
benchmark$descriptor[benchmark$descriptor == 'Attentivefp'] = 'Graph'

# Select the DL methods + a good and bad classical method
benchmark_dl = rbind(subset(benchmark, algorithm %in% c('GCN', 'MPNN', 'AFP', "GAT", 'DNN')),
                     subset(benchmark, algorithm == 'SVM' & descriptor == 'ECFP'),
                     subset(benchmark, algorithm == 'LSTM' & augmentation == 10),
                     subset(benchmark, algorithm == 'CNN' & augmentation == 10))


###### Boxplot #######


benchmark_box = benchmark_dl

# order factors of algorithm and descriptor
algo_order = c('GAT', 'GCN', 'MPNN', 'AFP', 'CNN', 'LSTM', 'DNN', "SVM")
benchmark_box$algorithm = factor(benchmark_box$algorithm, levels = algo_order)
benchmark_box$descriptor = factor(benchmark_box$descriptor, levels = c('ECFP', 'SMILES', 'Graph'))

colours = descr_cols$cols[match(c('ECFP', 'SMILES', 'Graph'), descr_cols$descr)]

# The plot
box_plot = ggplot(benchmark_box, aes(x=algorithm, y=cliff_rmse, fill=descriptor))+
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
  labs(x='Algorithm', y=bquote("RMSE"[cliff]))+
  guides(fill = 'none', 
         color = 'none')+
  default_theme

print(box_plot)


####### PCA ########

# Remove GAT. It is so bad that the whole plot shifts and becomes unreadable
benchmark_pca = subset(benchmark_dl, algorithm != 'GAT')

# Compute pca coordinates
pca_all = data_to_biplot(benchmark_pca, val_var="cliff_rmse")
bi_all = pca_all$bi

bi_all$algorithm = unlist(strsplit(as.character(bi_all$name), ' - '))[2*(1:length(bi_all$name))-1 ]
bi_all$descriptor = unlist(strsplit(as.character(bi_all$name), ' - '))[2*(1:length(bi_all$name)) ]

bi_all$algorithm[grepl('CHEMBL', bi_all$name)] = ''
bi_all$descriptor[grepl('CHEMBL', bi_all$name)] = ''

# rename some stuff
bi_all$algorithm = gsub('Svm', 'SVM', bi_all$algorithm)
bi_all$algorithm = gsub('Knn', 'KNN', bi_all$algorithm)
bi_all$algorithm = gsub('Cnn', 'CNN', bi_all$algorithm)
bi_all$algorithm = gsub('Dnn', 'DNN', bi_all$algorithm)
bi_all$algorithm = gsub('Lstm', 'LSTM', bi_all$algorithm)
bi_all$algorithm = gsub('Afp', 'AFP', bi_all$algorithm)
bi_all$algorithm = gsub('Gat', 'GAT', bi_all$algorithm)
bi_all$algorithm =  gsub('Gcn', 'GCN', bi_all$algorithm)
bi_all$algorithm =  gsub('Mpnn', 'MPNN', bi_all$algorithm)
bi_all$descriptor =  gsub('ecfp', 'ECFP', bi_all$descriptor)
bi_all$descriptor =  gsub('graph', 'Graph', bi_all$descriptor)
bi_all$descriptor =  gsub('smiles', 'SMILES', bi_all$descriptor)

# Define the colours for the descriptors
bi_all$col = descr_cols$cols[match(bi_all$descriptor, descr_cols$descr)]

# If 'Best' is on the left side of the plot, mirror everything. Makes it easer to compare with the previous plots
if (subset(bi_all, algorithm == 'Best')$x < 0){
  bi_all$x = bi_all$x*-1
}

# Get the coordinates of 'best' and 'worst' and find the best axis limit
best = unlist(subset(bi_all, algorithm == 'Best')[c(2,3)])
worst = unlist(subset(bi_all, algorithm == 'Worst')[c(2,3)])
axis_limit = ceiling(max(abs(subset(bi_all, type == 'Score')[c(2,3)]))) 

# Make the actual plot
pca_plot = ggplot(bi_all, aes(x = x, y =y)) +
  geom_hline(yintercept = 0, linetype = 'dashed', alpha = 0.25) +
  geom_vline(xintercept = 0, linetype = 'dashed', alpha = 0.25) +
  geom_segment(aes(x = worst[1], y = worst[2], xend = best[1], yend = best[2]),
               linetype='solid',  alpha = 0.005, size=0.75)+
  geom_point(aes(x, y ), colour = bi_all$col,  size = 1, shape=1,
             alpha = ifelse(bi_all$type == 'Score', 0.5, 0)) +
  
  geom_point(aes(x, y ), colour = bi_all$col,  size = 1, shape=19,
             alpha = ifelse(bi_all$type == 'Score', 0.5, 0)) +
  
  geom_text_repel(aes(label = algorithm), colour = bi_all$col,  
                  alpha = ifelse(bi_all$type == 'Score', 1, 0), 
                  size = 3, 
                  segment.size = 0.25, force = 10,
                  size=12, fontface="bold", max.iter = 1505, 
                  max.overlaps = 30, show.legend = FALSE)+

  scale_y_continuous(limits = c(-2.5, 2.5), expand = expansion(mult = c(0.01, 0.01)), breaks = seq(-3,3,1))+
  scale_x_continuous(limits = c(-2.5, 2.5), expand = expansion(mult = c(0.01, 0.01)), breaks = seq(-3,3,1))+
  coord_cartesian(ylim=c(-2, 2), xlim=c(-2.5, 2.5))+
  
  labs(x = paste0('PC ',1, ' (',round(pca_all$scree$data$eig[1],1),'%)'),
       y = paste0('PC ',2, ' (',round(pca_all$scree$data$eig[2],1),'%)'),
       shape = 'Algorithm',  color = 'Descriptor') +
  
  guides(color = guide_legend(override.aes = list(shape = 16), order = 1 )) +
  default_theme

print(pca_plot)


###### Scatter plots ######

benchmark_scatter = benchmark_dl
benchmark_scatter$descriptor = factor(benchmark_scatter$descriptor, levels = c('ECFP', 'SMILES', 'Graph'))
benchmark_scatter$algorithm = factor(benchmark_scatter$algorithm, levels = c('GCN', 'LSTM', 'DNN', 'SVM', 'AFP', 'CNN', 'GAT', 'MPNN')) 


gat_relative_plot = ggplot(benchmark_scatter, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'GAT', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'GAT', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'GAT', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  geom_text(x=0.75, y=2.25, label="GAT", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_x_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  scale_y_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  labs(x = '',  y = bquote("RMSE"[cliff])) +
  scale_color_manual(values = c(rep('#b6b6b6', 6), descr_cols$cols[which(descr_cols$descr == 'Graph')], rep('#b6b6b6', 1)))+
  coord_cartesian(ylim=c(0.5, 2.5), xlim=c(0.5, 2.5))+
  default_theme +
  theme(legend.position = 'none')

mpnn_relative_plot = ggplot(benchmark_scatter, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'MPNN', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'MPNN', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'MPNN', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  geom_text(x=0.75, y=2.25, label="MPNN", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_x_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  scale_y_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  labs(x = '',  y = '') +
  scale_color_manual(values = c(rep('#b6b6b6', 7), descr_cols$cols[which(descr_cols$descr == 'Graph')]))+
  coord_cartesian(ylim=c(0.5, 2.5), xlim=c(0.5, 2.5))+
  default_theme +
  theme(legend.position = 'none')

afp_relative_plot = ggplot(benchmark_scatter, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'AFP', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'AFP', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'AFP', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  geom_text(x=0.75, y=2.25, label="AFP", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_x_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  scale_y_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  labs(x = '',  y = '') +
  scale_color_manual(values = c(rep('#b6b6b6', 4), descr_cols$cols[which(descr_cols$descr == 'Graph')], rep('#b6b6b6', 3)))+
  coord_cartesian(ylim=c(0.5, 2.5), xlim=c(0.5, 2.5))+
  default_theme +
  theme(legend.position = 'none')

gcn_relative_plot = ggplot(benchmark_scatter, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'GCN', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'GCN', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'GCN', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  geom_text(x=0.75, y=2.25, label="GCN", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_x_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  scale_y_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  labs(x = '',  y = '') +
  scale_color_manual(values = c(descr_cols$cols[which(descr_cols$descr == 'Graph')], rep('#b6b6b6', 7)))+
  coord_cartesian(ylim=c(0.5, 2.5), xlim=c(0.5, 2.5))+
  default_theme +
  theme(legend.position = 'none')

cnn_relative_plot = ggplot(benchmark_scatter, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'CNN', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'CNN', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'CNN', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  geom_text(x=0.75, y=2.25, label="CNN", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_x_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  scale_y_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  labs(x = 'RMSE',  y = bquote("RMSE"[cliff])) +
  scale_color_manual(values = c(rep('#b6b6b6', 5), descr_cols$cols[which(descr_cols$descr == 'SMILES')], rep('#b6b6b6', 2)))+
  coord_cartesian(ylim=c(0.5, 2.5), xlim=c(0.5, 2.5))+
  default_theme +
  theme(legend.position = 'none')


lstm_relative_plot = ggplot(benchmark_scatter, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'LSTM', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'LSTM', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'LSTM', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  geom_text(x=0.75, y=2.25, label="LSTM", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_x_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  scale_y_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  labs(x = 'RMSE',  y = '') +
  scale_color_manual(values = c('#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'SMILES')], rep('#b6b6b6', 6)))+
  coord_cartesian(ylim=c(0.5, 2.5), xlim=c(0.5, 2.5))+
  default_theme +
  theme(legend.position = 'none')

dnn_relative_plot = ggplot(benchmark_scatter, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'DNN', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'DNN', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'DNN', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  geom_text(x=0.75, y=2.25, label="DNN", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_x_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  scale_y_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  labs(x = 'RMSE',  y = '') +
  scale_color_manual(values = c('#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'ECFP')], rep('#b6b6b6', 5)))+
  coord_cartesian(ylim=c(0.5, 2.5), xlim=c(0.5, 2.5))+
  default_theme +
  theme(legend.position = 'none')

svm_relative_plot = ggplot(benchmark_scatter, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'SVM', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'SVM', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark_scatter$algorithm == 'SVM', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  geom_text(x=0.75, y=2.25, label="SVM", color = 'black', size=2.5, fontface="bold", hjust=0) +
  scale_x_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  scale_y_continuous(breaks = seq(0,3,0.5), limits = c(0.01,3), expand = expansion(mult = c(0.01, 0.01)))+
  labs(x = 'RMSE',  y = '') +
  scale_color_manual(values = c('#b6b6b6', '#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'ECFP')], rep('#b6b6b6', 4)))+
  coord_cartesian(ylim=c(0.5, 2.5), xlim=c(0.5, 2.5))+
  default_theme +
  theme(legend.position = 'none')




##### Combine subplots ######

box_pca = plot_grid(box_plot, pca_plot, labels = c('a', 'b'), label_size=10, ncol=2, nrow=1, scale=1)

scatters = plot_grid(gat_relative_plot, gcn_relative_plot, mpnn_relative_plot, afp_relative_plot, 
                     cnn_relative_plot, lstm_relative_plot, dnn_relative_plot, svm_relative_plot, 
                     ncol=4, nrow=2, scale=1)
fig = plot_grid(box_pca, scatters, ncol=1, nrow =2, scale = 1, rel_heights=c(0.5, 0.6), labels = c('', 'c'), label_size = 10)

print(fig)
dev.print(pdf, 'Experiments/figures/Fig_4.pdf', width = 7.205, height = 6)


