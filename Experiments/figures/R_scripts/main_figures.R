#### Set working dir and source some functions ####

setwd("/Users/derekvantilborg/Dropbox/PycharmProjects/MoleculeACE")
source('Experiments/figures/R_scripts/vis_utils.R')

#### Data prep ####

# Import data
benchmark = read_csv('/Users/derekvantilborg/Dropbox/PycharmProjects/MoleculeACE/MoleculeACE/Data/results/MoleculeACE_results.csv')

# Rename some stuff
benchmark$descriptor[benchmark$descriptor == 'TOKENS'] = 'SMILES'
benchmark$descriptor[benchmark$descriptor == 'PHYSCHEM'] = 'Physchem'
benchmark$descriptor[benchmark$descriptor == 'MACCS'] = 'MACCs'
benchmark$descriptor[benchmark$descriptor == 'GRAPH'] = 'Graph'

#### Figures ####
figure3 = function(df){

  # Select only the classical methods
  df = subset(df, algorithm %in% c('RF', 'SVM', 'GBM', 'KNN'))
  
  # order factors of algorithm and descriptor
  mean_algo = df %>% group_by(algorithm) %>% summarise_all("mean")
  mean_descr = df %>% group_by(descriptor) %>% summarise_all("mean")
  
  algo_order = mean_algo[rev(order(mean_algo$cliff_rmse)),]$algorithm
  descr_order = mean_descr[rev(order(mean_descr$cliff_rmse)),]$descriptor
  
  # re-oder factors based on mean performance
  df$algorithm = factor(df$algorithm, levels = algo_order)
  df$descriptor = factor(df$descriptor, levels = descr_order)
  
  
  ### Boxplot ###
  colours = descr_cols$cols[match(c('WHIM', 'Physchem', 'MACCs', 'ECFP'), descr_cols$descr)]
  
  box_plot = ggplot(df, aes(x=algorithm, y=cliff_rmse, fill = descriptor))+
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=1, alpha=0.5) +
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=19, alpha=0.5) +
    geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.5,
                 outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=0.75) +
    scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)))+
    scale_color_manual(values = colours)+
    scale_fill_manual(values = colours)+
    labs(x='Algorithm', y=bquote("RMSE"[cliff]), fill = 'Descriptor')+
    guides(fill = 'none', color = 'none')+
    default_theme

  ### PCA ###
  pca_all = data_to_biplot(df, val_var="cliff_rmse" )
  bi_all = pca_all$bi
  
  # Fix some names
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
  
  x_axis_label = paste0('PC1 (',round(pca_all$scree$data$eig[1],1),'%)')
  y_axis_label = paste0('PC2 (',round(pca_all$scree$data$eig[2],1),'%)')
  
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
    scale_y_continuous(limits = c(-1.35, 1.35), expand = expansion(mult = c(0.01, 0.01)), breaks = seq(-2,2,1))+
    scale_x_continuous(limits = c(-1.35, 1.35), expand = expansion(mult = c(0.01, 0.01)), breaks = seq(-2,2,1))+
    coord_cartesian(ylim=c(-1.1, 1.1))+
    labs(x = x_axis_label, y = y_axis_label, shape = 'Algorithm',  color = 'Descriptor') +
    guides(shape = guide_legend(override.aes = list(color = "#27275d"), order = 2),
           color = guide_legend(override.aes = list(shape = 16), order = 1 )) +
    default_theme

  ### Scatter plots ###
  relative_scatter = function(descriptor, colours){
    p = ggplot(df, aes(x=rmse, y=cliff_rmse, colour=descriptor))+
      geom_point(alpha = ifelse(df$descriptor == descriptor, 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
      geom_point(alpha = ifelse(df$descriptor == descriptor, 0.5, 0), shape=1, size=1)+
      geom_point(alpha = ifelse(df$descriptor == descriptor, 0.5, 0), shape=19, size=1)+
      geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
      geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
      geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
      scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01))) +
      scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01))) +
      labs(x = 'RMSE',  y = '') +
      scale_color_manual(values = colours)+
      default_theme +
      theme(legend.position = 'none')
    
    return(p)
  }
  
  ecfp4_relative_plot = relative_scatter('ECFP', c('#b6b6b6', '#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'ECFP')]))
  maccs_relative_plot = relative_scatter('MACCs', c('#b6b6b6', '#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'MACCs')], '#b6b6b6'))
  physchem_relative_plot = relative_scatter('Physchem', c('#b6b6b6', descr_cols$cols[which(descr_cols$descr == 'Physchem')], '#b6b6b6', '#b6b6b6'))
  whim_relative_plot = relative_scatter('WHIM', c(descr_cols$cols[which(descr_cols$descr == 'WHIM')], '#b6b6b6', '#b6b6b6', '#b6b6b6'))

  ### combine subplots ###
  box_pca = plot_grid(box_plot, pca_plot, labels = c('a', 'b'), label_size = 10, ncol=2, nrow =1, scale = 1)
  scatters = plot_grid(whim_relative_plot, physchem_relative_plot, maccs_relative_plot, ecfp4_relative_plot, ncol=4, nrow =1, scale = 1)
  fig = plot_grid(box_pca, scatters, ncol=1, nrow =2, scale = 1, rel_heights=c(0.6, 0.4), labels = c('', 'c'), label_size = 10)

  return(fig)
}


figure4 = function(df){
  
  # Select the DL methods + a good and bad classical method
  df = rbind(subset(df, algorithm %in% c('GCN', 'MPNN', 'AFP', "GAT", 'MLP', 'Transformer')),
                       # subset(df, algorithm == 'SVM' & descriptor == 'ECFP'),
                       subset(df, algorithm == 'LSTM' & augmentation == 10),
                       subset(df, algorithm == 'CNN' & augmentation == 10))
  
  # order factors of algorithm and descriptor
  algo_order = c('GAT', 'GCN', 'AFP', 'MPNN', 'CNN', 'Transformer', 'LSTM', 'MLP', "SVM")
  df$algorithm = factor(df$algorithm, levels = algo_order)
  df$descriptor = factor(df$descriptor, levels = c('ECFP', 'SMILES', 'Graph'))
  
  colours = descr_cols$cols[match(c('ECFP', 'SMILES', 'Graph'), descr_cols$descr)]
  
  
  ### Box plot ###

  box_plot = ggplot(df, aes(x=algorithm, y=cliff_rmse, fill=descriptor))+
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=1, alpha=0.5) +
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=19, alpha=0.5) +
    geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
                 outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=1) +
    scale_y_continuous(breaks = seq(0.25,1.75,0.25), expand = expansion(mult = c(0.01, 0.01)))+
    coord_cartesian(ylim=c(0.25, 1.75))+
    scale_color_manual(values = colours)+
    scale_fill_manual(values = colours)+
    labs(x='Algorithm', y=bquote("RMSE"[cliff]))+
    guides(fill = 'none', 
           color = 'none')+
    default_theme
  

  ### PCA  ###

  # Compute pca coordinates
  pca_all = data_to_biplot(df, val_var="cliff_rmse")
  bi_all = pca_all$bi
  
  bi_all$algorithm = unlist(strsplit(as.character(bi_all$name), ' - '))[2*(1:length(bi_all$name))-1 ]
  bi_all$descriptor = unlist(strsplit(as.character(bi_all$name), ' - '))[2*(1:length(bi_all$name)) ]
  bi_all$algorithm[grepl('CHEMBL', bi_all$name)] = ''
  bi_all$descriptor[grepl('CHEMBL', bi_all$name)] = ''
  
  # rename some stuff
  bi_all$algorithm = gsub('Svm', 'SVM', bi_all$algorithm)
  bi_all$algorithm = gsub('Knn', 'KNN', bi_all$algorithm)
  bi_all$algorithm = gsub('Cnn', 'CNN', bi_all$algorithm)
  bi_all$algorithm = gsub('Mlp', 'MLP', bi_all$algorithm)
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
    geom_segment(aes(x = worst[1], y = worst[2], xend = best[1], yend = best[2]), linetype='solid',  alpha = 0.005, size=0.75)+
    geom_point(aes(x, y ), colour = bi_all$col,  size = 1, shape=1, alpha = ifelse(bi_all$type == 'Score', 0.5, 0)) +
    geom_point(aes(x, y ), colour = bi_all$col,  size = 1, shape=19, alpha = ifelse(bi_all$type == 'Score', 0.5, 0)) +
    geom_text_repel(aes(label = algorithm), colour = bi_all$col,   alpha = ifelse(bi_all$type == 'Score', 1, 0), 
                    size = 3, segment.size = 0.25, force = 10, size=12, fontface="bold", max.iter = 1505, 
                    max.overlaps = 30, show.legend = FALSE)+
    scale_y_continuous(limits = c(-2.5, 2.5), expand = expansion(mult = c(0.01, 0.01)), breaks = seq(-3,3,1))+
    scale_x_continuous(limits = c(-2.5, 2.5), expand = expansion(mult = c(0.01, 0.01)), breaks = seq(-3,3,1))+
    coord_cartesian(ylim=c(-1.1, 1.1), xlim=c(-1.4, 1.4))+
    labs(x = paste0('PC ',1, ' (',round(pca_all$scree$data$eig[1],1),'%)'),
         y = paste0('PC ',2, ' (',round(pca_all$scree$data$eig[2],1),'%)'),
         shape = 'Algorithm',  color = 'Descriptor') +
    guides(color = guide_legend(override.aes = list(shape = 16), order = 1 )) +
    default_theme
  
    
    ### Scatter plot s###

  df$descriptor = factor(df$descriptor, levels = c('ECFP', 'SMILES', 'Graph'))
  df$algorithm = factor(df$algorithm, levels = c('GCN', 'LSTM', 'MLP', 'SVM', 'AFP', 'CNN', 'GAT', 'MPNN', 'Transformer')) 
  
  scatter_plot = function(algo, descriptor){
    
    colours = c(rep('#b6b6b6', length(unique(df$algorithm))))
    colours[which(levels(df$algorithm) == algo)] = descr_cols$cols[which(descr_cols$descr == descriptor)]
    
    p = ggplot(df, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
      geom_point(alpha = ifelse(df$algorithm == algo, 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
      geom_point(alpha = ifelse(df$algorithm == algo, 0.5, 0), shape=1, size=1)+
      geom_point(alpha = ifelse(df$algorithm == algo, 0.5, 0), shape=19, size=1)+
      geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
      geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
      geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
      geom_text(x=0.35, y=1.6, label=algo, color = 'black', size=2.5, fontface="bold", hjust=0) +
      scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.01,1.75), expand = expansion(mult = c(0.01, 0.01)))+
      scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.01,1.75), expand = expansion(mult = c(0.01, 0.01)))+
      labs(x = '',  y = bquote("RMSE"[cliff])) +
      scale_color_manual(values = colours)+
      coord_cartesian(ylim=c(0.25, 1.75), xlim=c(0.25, 1.75))+
      default_theme +
      theme(legend.position = 'none')
    
    return(p)
  }
  
  
  gat_relative_plot = scatter_plot('GAT', 'Graph')
  mpnn_relative_plot = scatter_plot('MPNN', 'Graph')
  afp_relative_plot = scatter_plot('AFP', 'Graph')
  gcn_relative_plot = scatter_plot('GCN', 'Graph')
  cnn_relative_plot = scatter_plot('CNN', 'SMILES')
  trans_relative_plot = scatter_plot('Transformer', 'SMILES')
  lstm_relative_plot = scatter_plot('LSTM', 'SMILES')
  mlp_relative_plot= scatter_plot('MLP', 'ECFP')
  
  
  ### combine all plots ###
  box_pca = plot_grid(box_plot, pca_plot, labels = c('a', 'b'), label_size=10, ncol=2, nrow=1, scale=1)
  scatters = plot_grid(gat_relative_plot, gcn_relative_plot, afp_relative_plot, mpnn_relative_plot, 
                       cnn_relative_plot, trans_relative_plot, lstm_relative_plot, mlp_relative_plot, 
                       ncol=4, nrow=2, scale=1)
  fig = plot_grid(box_pca, scatters, ncol=1, nrow =2, scale = 1, rel_heights=c(0.4, 0.6), labels = c('', 'c'), label_size = 10)
  
  return(fig)
  
}


figure5 = function(df){

  # add dataset abbreviation
  df$label = dataset_abbrv$abbrv[match(df$dataset, dataset_abbrv$dataset)]
  # Calculate RMSE delta
  df$rmse_delta = df$cliff_rmse - df$rmse
  
  # Order datasets
  mean_dataset <- df %>% group_by(label) %>%  summarise(MinDataset = cor(cliff_rmse, rmse))
  dataset_order = mean_dataset$label[order(-mean_dataset$MinDataset)]
  df$label = factor(df$label, levels = dataset_order)
  
  df$corr = mean_dataset$MinDataset[match(df$label, mean_dataset$label)]
  df$corr_label = as.character(round(df$corr,2))
  
  df$descriptor = factor(df$descriptor, levels = c("Graph","ECFP","MACCs","Physchem","SMILES","WHIM"))
  colours = descr_cols$cols[match(levels(df$descriptor), descr_cols$descr)]
  
  # Plot RMSE RMSEcliff scatters
  p_good = ggplot(subset(df, dataset == "CHEMBL214_Ki"), aes(x = rmse, y = cliff_rmse))+
    geom_point(size=1, shape=1, alpha=0.75, aes(color = descriptor))+
    geom_point(size=1, shape=19, alpha=0.75, aes(color = descriptor))+
    geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
    geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
    geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
    labs(x='', y=bquote("RMSE"[cliff]), fill = 'Descriptor')+
    scale_color_manual(values = colours)+
    scale_fill_manual(values = colours)+
    guides(fill = 'none', color = 'none')+
    scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.01,1.75), expand = expansion(mult = c(0.01, 0.01)))+
    scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.01,1.75), expand = expansion(mult = c(0.01, 0.01)))+
    coord_cartesian(ylim=c(0.25, 1.75), xlim=c(0.25, 1.75))+
    theme(
      panel.border = element_rect(colour = "black", size = 1, fill = NA),
      panel.background = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "plain"),
      axis.ticks.y = element_line(colour = "black"),
      axis.ticks.x = element_line(colour = "black"),
      axis.text.y = element_text(size=6, face="plain", colour = "black"),
      axis.text.x = element_text(size=6, face="plain", colour = "black"),
      axis.title.x = element_text(size=6, face="plain", colour = "black"),
      axis.title.y = element_text(size=6, face="plain", colour = "black"),
      legend.key = element_blank(),
      legend.text = element_text(colour = "black"),
      legend.position = 'right', legend.box = "vertical",
      legend.title = element_blank(),
      legend.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank())

  p_bad = ggplot(subset(df, dataset == "CHEMBL4203_Ki"), aes(x = rmse, y = cliff_rmse))+
    geom_point(size=1, shape=1, alpha=0.75, aes(color = descriptor))+
    geom_point(size=1, shape=19, alpha=0.75, aes(color = descriptor))+
    geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
    geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
    geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
    labs(x='RMSE', y=bquote("RMSE"[cliff]), fill = 'Descriptor')+
    scale_color_manual(values = colours)+
    scale_fill_manual(values = colours)+
    guides(fill = 'none', color = 'none')+
    scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.01,1.75), expand = expansion(mult = c(0.01, 0.01)))+
    scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.01,1.75), expand = expansion(mult = c(0.01, 0.01)))+
    coord_cartesian(ylim=c(0.25, 1.75), xlim=c(0.25, 1.75))+
    theme(
      panel.border = element_rect(colour = "black", size = 1, fill = NA),
      panel.background = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "plain"),
      axis.ticks.y = element_line(colour = "black"),
      axis.ticks.x = element_line(colour = "black"),
      axis.text.y = element_text(size=6, face="plain", colour = "black"),
      axis.text.x = element_text(size=6, face="plain", colour = "black"),
      axis.title.x = element_text(size=6, face="plain", colour = "black"),
      axis.title.y = element_text(size=6, face="plain", colour = "black"),
      legend.key = element_blank(),
      legend.text = element_text(colour = "black"),
      legend.position = 'right', legend.box = "vertical",
      legend.title = element_blank(),
      legend.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank())

  # Make plot
  diff_per_ds = ggplot(df, aes(x=rmse_delta, y=label, fill = descriptor))+
    geom_vline(xintercept = 0, linetype='dashed', alpha=0.5, colour='#27275d')+
    stat_boxplot(geom ='errorbar', coef=NULL, aes(group=dataset), lwd=0.25, width = 0.5, size=0.5) +
    geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.5,
                 outlier.shape=NA, varwidth = FALSE, lwd=0.25, fatten=2, aes(group=dataset), fill='#27275d', colour='#27275d') +
    scale_y_discrete(position = "left") +
    scale_x_continuous(labels = c('-0.2', '0', '0.2', '0.4', '0.6')) +
    labs(y='Dataset', x=bquote("RMSE"[cliff]~"- RMSE"), fill = 'Descriptor')+
    guides(fill = 'none', color = 'none')+
    coord_cartesian(xlim=c(-0.2, 0.6))+
    default_theme +
    theme(
      panel.border = element_rect(colour = "black", size = 1, fill = NA),
      panel.background = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "plain"),
      axis.ticks.y = element_line(colour = "black"),
      axis.ticks.x = element_line(colour = "black"),
      axis.text.y = element_text(size=6, face="plain", colour = "black"),
      axis.text.x = element_text(size=6, face="plain", colour = "black"),
      axis.title.x = element_text(size=6, face="plain", colour = "black"),
      axis.title.y = element_text(size=6, face="plain", colour = "black"),
      legend.key = element_blank(),
      legend.text = element_text(colour = "black"),
      legend.position = 'right', legend.box = "vertical",
      legend.title = element_blank(),
      legend.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = unit(c(0, 0, 0, 0), "cm"))
  
  
  corr_per_label = df %>% group_by(label) %>% summarize(cor=mean(corr))
  corr_per_label$label = factor(corr_per_label$label, levels = dataset_order)
  corr_per_label$cor_label = as.character(round(corr_per_label$cor,2))
  
  corr_bar = ggplot(corr_per_label, aes(y=label, x=1, fill=cor))+
    geom_tile()+
    geom_text_repel(aes(x = 1, label = cor_label), size = 1.5, 
                    segment.size = 0.25, force = 0, fontface="bold", max.iter = 10, 
                    max.overlaps = 1000, show.legend = FALSE, color='white')+
    guides(fill = 'none', color = 'none')+
    theme(
      panel.border = element_blank(),
      panel.background = element_blank(),
      plot.title = element_blank(),
      axis.ticks.y = element_blank(),
      axis.ticks.x = element_blank(),
      axis.text.y = element_blank(),
      axis.text.x = element_blank(),
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(colour = "black"),
      legend.position = 'right', legend.box = "vertical",
      legend.title = element_blank(),
      legend.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = unit(c(0, 0, 0.5, 0), "cm"))


  ## Train size effects
  datasets <- read_csv("MoleculeACE/Data/benchmark_data/metadata/datasets.csv")
  df$train_size = datasets$`Train compounds`[match(df$dataset, datasets$Dataset)]
  corr_per_dataset = df %>% group_by(dataset) %>% summarize(cor=cor(rmse, cliff_rmse))
  corr_per_dataset$train_size = datasets$`Train compounds`[match(corr_per_dataset$dataset, datasets$Dataset)]
  
  p_corr = ggplot(corr_per_dataset, aes(y = cor, x = train_size))+
    geom_point(size=1, shape=1, alpha=0.75, colour='#27275d') +
    geom_point(size=1, shape=19, alpha=0.75, colour='#27275d') +  #'
    labs(x="Train molecules", y=bquote("r(RMSE, RMSE"[cliff]~")")) +
    scale_x_continuous(breaks = seq(500,3000,500), limits = c(450,3000), expand = expansion(mult = c(0.01, 0.01))) +
    default_theme


  rmse_per_dataset <- group_by(df, dataset)
  rmse_per_dataset = summarise(rmse_per_dataset, mean_rmse_delta=mean(rmse_delta), min_rmse_delta=min(rmse_delta),
                               max_rmse_delta=max(rmse_delta), train_size=mean(train_size) )

  p_datasize = ggplot(rmse_per_dataset, aes(y = mean_rmse_delta, x = train_size))+
    geom_errorbar(aes(ymin=min_rmse_delta, ymax=max_rmse_delta), 
                  colour='#27275d', width=0, size=0.5, alpha=0.5)+
    geom_point(size=1, shape=1, alpha=0.75, colour='#27275d') +
    geom_point(size=1, shape=19, alpha=0.75, colour='#27275d') +
    labs(x="Train molecules", y=bquote("RMSE"[cliff]~"- RMSE")) +
    scale_x_continuous(breaks = seq(500,3000,500), limits = c(450,3000), expand = expansion(mult = c(0.01, 0.01))) +
    coord_cartesian(ylim=c(-0.2, 0.6))+
    scale_color_manual(values = colours)+
    guides(fill = 'none', color = 'none')+
    default_theme
  
  
  p_data =plot_grid(p_datasize, p_corr,
                      label_size = 10, labels = c('d', 'e'),
                      ncol=2, nrow =1, scale = 1)
  
  scatters = plot_grid(p_good, p_bad,
                       label_size = 10, labels = c('b', 'c'),
                       ncol=1, nrow =2, scale = 1)
  
  plot_corbar = plot_grid(diff_per_ds, corr_bar, ncol=2, nrow =1, scale = 1,
                          rel_widths = c(0.85, 0.15))
  
  scatters = plot_grid(plot_corbar, scatters,
                       label_size = 10, labels = c('a', ''),
                       ncol=2, nrow =1, scale = 1)
  
  scatters = plot_grid(scatters, p_data,
                       label_size = 10, labels = c('', ''),
                       ncol=1, nrow =2, scale = 1, rel_heights=c(0.666, 0.333))
  
  return(scatters)

}


print(figure3(benchmark))
dev.print(pdf, 'Experiments/figures/Fig_3.pdf', width = 7.205, height = 4)

print(figure4(benchmark))
dev.print(pdf, 'Experiments/figures/Fig_4.pdf', width = 7.205, height = 5.2)

print(figure5(benchmark))
dev.print(pdf, 'Experiments/figures/Fig_5.pdf', width = 3.504, height = 5)
