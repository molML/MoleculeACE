###### Set working dir and source some functions #######


setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")
source('Experiments/figures/R_scripts/vis_utils.R')


###### Data prep #######


# Import data
benchmark = read_csv('Results/Benchmark_results.csv')

# Select only the classical methods
benchmark = subset(benchmark, algorithm %in% c('RF', 'SVM', 'GBM', 'KNN'))

# Rename some stuff
names(benchmark) = gsub('ac-rmse_soft_consensus','cliff_rmse',names(benchmark))

#### All subplots


scatter_mini_plot = function(data, algo, descr, manual_cols, xlab='', ylab='', label=''){
  
  ggplot(data, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
    geom_point(alpha = ifelse(data$algorithm == algo & data$descriptor == descr, 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
    geom_point(alpha = ifelse(data$algorithm == algo & data$descriptor == descr, 0.5, 0), shape=1, size=1)+
    geom_point(alpha = ifelse(data$algorithm == algo & data$descriptor == descr, 0.5, 0), shape=19, size=1)+
    geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
    geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
    geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
    geom_text(x=0.5, y=1.5, label=label, color = 'black', size=2.5, fontface="bold", hjust=0) +
    scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
    scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)), labels = c('','0.5','','1','','1.5','')) +
    labs(x = xlab,  y = ylab) +
    scale_color_manual(values = manual_cols)+
    default_theme +
    theme(legend.position = 'none')
}


whim_1 = scatter_mini_plot(benchmark, 'KNN', 'WHIM', c(descr_cols$cols[which(descr_cols$descr == 'WHIM')], '#b6b6b6', '#b6b6b6', '#b6b6b6'),
                           xlab='', ylab=bquote("RMSE"[cliff]), label='KNN')
whim_2 = scatter_mini_plot(benchmark, 'GBM', 'WHIM', c('#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'WHIM')], '#b6b6b6'),
                           xlab='', ylab='', label='GBM')
whim_3 = scatter_mini_plot(benchmark, 'RF', 'WHIM', c('#b6b6b6', '#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'WHIM')]),
                           xlab='', ylab='', label='RF')
whim_4 = scatter_mini_plot(benchmark, 'SVM', 'WHIM', c('#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'WHIM')], '#b6b6b6', '#b6b6b6'),
                           xlab='', ylab='', label='SVM')


phys_1 = scatter_mini_plot(benchmark, 'KNN', 'Physchem', c(descr_cols$cols[which(descr_cols$descr == 'Physchem')], '#b6b6b6', '#b6b6b6', '#b6b6b6'),
                           xlab='', ylab=bquote("RMSE"[cliff]), label='KNN')
phys_2 = scatter_mini_plot(benchmark, 'GBM', 'Physchem', c('#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'Physchem')], '#b6b6b6'),
                           xlab='', ylab='', label='GBM')
phys_3 = scatter_mini_plot(benchmark, 'RF', 'Physchem', c('#b6b6b6', '#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'Physchem')]),
                           xlab='', ylab='', label='RF')
phys_4 = scatter_mini_plot(benchmark, 'SVM', 'Physchem', c('#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'Physchem')], '#b6b6b6', '#b6b6b6'),
                           xlab='', ylab='', label='SVM')


maccs_1 = scatter_mini_plot(benchmark, 'KNN', 'MACCs', c(descr_cols$cols[which(descr_cols$descr == 'MACCs')], '#b6b6b6', '#b6b6b6', '#b6b6b6'),
                            xlab='', ylab=bquote("RMSE"[cliff]), label='KNN')
maccs_2 = scatter_mini_plot(benchmark, 'GBM', 'MACCs', c('#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'MACCs')], '#b6b6b6'),
                            xlab='', ylab='', label='GBM')
maccs_3 = scatter_mini_plot(benchmark, 'RF', 'MACCs', c('#b6b6b6', '#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'MACCs')]),
                            xlab='', ylab='', label='RF')
maccs_4 = scatter_mini_plot(benchmark, 'SVM', 'MACCs', c('#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'MACCs')], '#b6b6b6', '#b6b6b6'),
                            xlab='', ylab='', label='SVM')


ecfp_1 = scatter_mini_plot(benchmark, 'KNN', 'ECFP', c(descr_cols$cols[which(descr_cols$descr == 'ECFP')], '#b6b6b6', '#b6b6b6', '#b6b6b6'),
                           xlab='RMSE', ylab=bquote("RMSE"[cliff]), label='KNN')
ecfp_2 = scatter_mini_plot(benchmark, 'GBM', 'ECFP', c('#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'ECFP')], '#b6b6b6'),
                           xlab='RMSE', ylab='', label='GBM')
ecfp_3 = scatter_mini_plot(benchmark, 'RF', 'ECFP', c('#b6b6b6', '#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'ECFP')]),
                           xlab='RMSE', ylab='', label='RF')
ecfp_4 = scatter_mini_plot(benchmark, 'SVM', 'ECFP', c('#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'ECFP')], '#b6b6b6', '#b6b6b6'),
                           xlab='RMSE', ylab='', label='SVM')


scatters = plot_grid(whim_1, whim_2, whim_3, whim_4,
                     phys_1, phys_2, phys_3, phys_4,
                     maccs_1, maccs_2, maccs_3, maccs_4,
                     ecfp_1, ecfp_2, ecfp_3, ecfp_4,
                     ncol=4, nrow =4, scale = 1)

print(scatters)
dev.print(pdf, 'Experiments/figures/sup_fig_3.pdf', width = 7.205, height = 7.205)
