###### Set working dir and source some functions #######


setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")
source('Experiments/figures/R_scripts/vis_utils.R')


###### Data prep #######


# Import data
benchmark = read_csv('Results/Benchmark_results.csv')

# Select only the classical methods
benchmark = subset(benchmark, algorithm %in% c('RF', 'SVM', 'GBM', 'KNN'))

# order factors of algorithm and descriptor
mean_algo = benchmark %>% group_by(algorithm) %>% summarise_all("mean")
mean_descr = benchmark %>% group_by(descriptor) %>% summarise_all("mean")

algo_order = mean_algo[rev(order(mean_algo$cliff_rmse)),]$algorithm
descr_order = mean_descr[rev(order(mean_descr$cliff_rmse)),]$descriptor

# re-oder factors based on mean performance
benchmark$algorithm = factor(benchmark$algorithm, levels = algo_order)
benchmark$descriptor = factor(benchmark$descriptor, levels = descr_order)


###### Boxplot #######


colours = descr_cols$cols[match(c('WHIM', 'Physchem', 'MACCs', 'ECFP'), descr_cols$descr)]

box_plot = ggplot(benchmark, aes(x=algorithm, y=cliff_rmse, fill = descriptor))+
  
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
  labs(x='Algorithm', y=bquote("RMSE"[cliff]), fill = 'Descriptor')+
  guides(fill = 'none', color = 'none')+
  default_theme

print(box_plot)


###### PCA #######


pca_dat = benchmark
pca_all = data_to_biplot(pca_dat, val_var="cliff_rmse" )
bi_all = pca_all$bi

bi_all$algorithm = unlist(strsplit(as.character(bi_all$name),  ' - '))[2*(1:length(bi_all$name))-1 ]
bi_all$descriptor = unlist(strsplit(as.character(bi_all$name),  ' - '))[2*(1:length(bi_all$name)) ]

bi_all$algorithm[grepl('CHEMBL', bi_all$name)] = ''
bi_all$descriptor[grepl('CHEMBL', bi_all$name)] = ''

# rename some stuff
bi_all$algorithm = gsub('Svm', 'SVM', bi_all$algorithm)
bi_all$algorithm = gsub('Knn', 'KNN', bi_all$algorithm)
bi_all$algorithm = gsub('Gbm', 'GBM', bi_all$algorithm)
bi_all$algorithm = gsub('Rf', 'RF', bi_all$algorithm)
bi_all$descriptor =  gsub('maccs', 'MACCs', bi_all$descriptor)
bi_all$descriptor =  gsub('physchem', 'Physchem', bi_all$descriptor)
bi_all$descriptor =  gsub('whim', 'WHIM', bi_all$descriptor)
bi_all$descriptor =  gsub('ecfp', 'ECFP', bi_all$descriptor)

# Give colours to the data
bi_all$col = descr_cols$cols[match(bi_all$descriptor, descr_cols$descr)]

# Get xy the coordinates for the best and worst points
best = unlist(subset(bi_all, algorithm == 'Best')[c(2,3)])
worst = unlist(subset(bi_all, algorithm == 'Worst')[c(2,3)])

# Make the actual plot
pca_plot = ggplot(bi_all, aes(x = x, y =y)) +
  
  geom_hline(yintercept = 0, linetype = 'dashed', alpha = 0.25) +
  geom_vline(xintercept = 0, linetype = 'dashed', alpha = 0.25) +
  geom_segment(aes(x = worst[1], y = worst[2], xend = best[1], yend = best[2]),
               linetype='solid',  alpha = 0.005, colour='#27275d', size=0.75)+
  
  geom_point(aes(x, y ), colour = bi_all$col, shape = 1,  size = 1, alpha = ifelse(bi_all$type == 'Score', 0.5, 0)) +
  geom_point(aes(x, y ), colour = bi_all$col, shape = 19,  size = 1, alpha = ifelse(bi_all$type == 'Score', 0.5, 0)) +
  
  geom_text_repel(aes(label = algorithm), colour = bi_all$col, alpha = ifelse(bi_all$type == 'Score', 1, 0), 
                  size = 3, segment.size = 0.25, force = 20, fontface="bold", max.iter = 1505, 
                  max.overlaps = 30, show.legend = FALSE)+
  
  scale_y_continuous(limits = c(-1.5, 1.5), expand = expansion(mult = c(0.01, 0.01)), breaks = seq(-2,2,1))+
  scale_x_continuous(limits = c(-1.5, 1.5), expand = expansion(mult = c(0.01, 0.01)), breaks = seq(-2,2,1))+
  coord_cartesian(ylim=c(-1.2, 1.2))+

  labs(x = paste0('PC1 (',round(pca_all$scree$data$eig[1],1),'%)'),
       y = paste0('PC2 (',round(pca_all$scree$data$eig[2],1),'%)'),
       shape = 'Algorithm',  color = 'Descriptor') +

  guides(shape = guide_legend(override.aes = list(color  = "#27275d"), order = 2),
         color = guide_legend(override.aes = list(shape = 16), order = 1 )) +
  default_theme

print(pca_plot)


##### Relative performance ######

# Cliff rmse vs normal rmse scatter plot
ecfp4_relative_plot = ggplot(benchmark, aes(x=rmse, y=cliff_rmse, colour=descriptor))+
  geom_point(alpha = ifelse(benchmark$descriptor == 'ECFP', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark$descriptor == 'ECFP', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark$descriptor == 'ECFP', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  labs(x = 'RMSE',  y = '') +
  scale_color_manual(values = c('#b6b6b6', '#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'ECFP')]))+
  default_theme +
  theme(legend.position = 'none')

maccs_relative_plot = ggplot(benchmark, aes(x=rmse, y=cliff_rmse, colour=descriptor))+
  geom_point(alpha = ifelse(benchmark$descriptor == 'MACCs', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark$descriptor == 'MACCs', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark$descriptor == 'MACCs', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  labs(x = 'RMSE',  y = '') +
  scale_color_manual(values = c('#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'MACCs')], '#b6b6b6'))+
  default_theme +
  theme(legend.position = 'none')

physchem_relative_plot = ggplot(benchmark, aes(x=rmse, y=cliff_rmse, colour=descriptor))+
  geom_point(alpha = ifelse(benchmark$descriptor == 'Physchem', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark$descriptor == 'Physchem', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark$descriptor == 'Physchem', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  labs(x = 'RMSE',  y = '') +
  scale_color_manual(values = c('#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'Physchem')], '#b6b6b6', '#b6b6b6'))+
  default_theme +
  theme(legend.position = 'none')

whim_relative_plot = ggplot(benchmark, aes(x=rmse, y=cliff_rmse, colour=descriptor))+
  geom_point(alpha = ifelse(benchmark$descriptor == 'WHIM', 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
  geom_point(alpha = ifelse(benchmark$descriptor == 'WHIM', 0.5, 0), shape=1, size=1)+
  geom_point(alpha = ifelse(benchmark$descriptor == 'WHIM', 0.5, 0), shape=19, size=1)+
  geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
  geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
  geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
  scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
  labs(x = 'RMSE',  y = bquote("RMSE"[cliff])) +
  scale_color_manual(values = c(descr_cols$cols[which(descr_cols$descr == 'WHIM')], '#b6b6b6', '#b6b6b6', '#b6b6b6'))+
  default_theme +
  theme(legend.position = 'none')


##### Combine subplots ######


box_pca = plot_grid(box_plot, pca_plot, labels = c('a', 'b'), label_size = 10, ncol=2, nrow =1, scale = 1)
scatters = plot_grid(whim_relative_plot, physchem_relative_plot, maccs_relative_plot, ecfp4_relative_plot, ncol=4, nrow =1, scale = 1)

fig = plot_grid(box_pca, scatters, ncol=1, nrow =2, scale = 1, rel_heights=c(0.625, 0.375), labels = c('', 'c'), label_size = 10)

print(fig)
dev.print(pdf, 'Experiments/figures/Fig_3.pdf', width = 7.205, height = 4.50)
