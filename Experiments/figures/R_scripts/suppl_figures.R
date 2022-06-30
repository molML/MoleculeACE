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
figure_s1 = function(df){
  
  # Select data
  df_classical = subset(df, algorithm %in% c('RF', 'SVM', 'GBM', 'KNN'))
  df_dl = rbind(subset(df, algorithm %in% c('GCN', 'MPNN', 'AFP', "GAT", 'MLP', 'Transformer')),
                subset(df, algorithm == 'SVM' & descriptor == 'ECFP'),
                subset(df, algorithm == 'LSTM' & augmentation == 10),
                subset(df, algorithm == 'CNN' & augmentation == 10))
  
  # order factors of algorithm and descriptor
  mean_algo = df_classical %>% group_by(algorithm) %>% summarise_all("mean")
  mean_descr = df_classical %>% group_by(descriptor) %>% summarise_all("mean")
  
  algo_order = mean_algo[rev(order(mean_algo$cliff_rmse)),]$algorithm
  descr_order = mean_descr[rev(order(mean_descr$cliff_rmse)),]$descriptor
  
  # re-oder factors based on mean performance
  df_classical$algorithm = factor(df_classical$algorithm, levels = algo_order)
  df_classical$descriptor = factor(df_classical$descriptor, levels = descr_order)
  

  # order factors of algorithm and descriptor
  algo_order = c('GAT', 'GCN', 'AFP', 'MPNN', 'CNN', 'Transformer', 'LSTM', 'MLP', "SVM")
  df_dl$algorithm = factor(df_dl$algorithm, levels = algo_order)
  df_dl$descriptor = factor(df_dl$descriptor, levels = c('ECFP', 'SMILES', 'Graph'))
  
  colours_classical = descr_cols$cols[match(c('WHIM', 'Physchem', 'MACCs', 'ECFP'), descr_cols$descr)]
  colours_dl = descr_cols$cols[match(c('ECFP', 'SMILES', 'Graph'), descr_cols$descr)]
  
  box_plot_classical = ggplot(df_classical, aes(x=algorithm, y=rmse, fill = descriptor))+
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=1, alpha=0.5) +
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=19, alpha=0.5) +
    geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.5,
                 outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=0.75) +
    scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)))+
    coord_cartesian(ylim=c(0.25, 1.75))+
    scale_color_manual(values = colours_classical)+
    scale_fill_manual(values = colours_classical)+
    labs(x='Algorithm', y="RMSE", fill = 'Descriptor')+
    guides(fill = 'none', color = 'none')+
    default_theme

  box_plot_dl = ggplot(df_dl, aes(x=algorithm, y=rmse, fill=descriptor))+
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=1, alpha=0.5) +
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=19, alpha=0.5) +
    geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
                 outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=1) +
    scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)))+
    coord_cartesian(ylim=c(0.25, 1.75))+
    scale_color_manual(values = colours_dl)+
    scale_fill_manual(values = colours_dl)+
    labs(x='Algorithm', y=bquote("RMSE"[cliff]))+
    guides(fill = 'none', 
           color = 'none')+
    default_theme
  
  box = plot_grid(box_plot_classical, box_plot_dl, labels = c('a', 'b'), label_size=10, ncol=2, nrow=1, scale=1)
  
  return(box)
}


figure_s2 = function(df){
  
  # Select only the classical methods
  benchmark_ml = subset(df, algorithm %in% c('RF', 'SVM', 'GBM', 'KNN'))
  # Select the DL methods + a good and bad classical method
  benchmark_dl = rbind(subset(df, algorithm %in% c('GCN', 'MPNN', 'AFP', "GAT", 'MLP', 'Transformer')),
                       subset(df, algorithm == 'SVM' & descriptor == 'ECFP'),
                       subset(df, algorithm == 'LSTM' & augmentation == 10),
                       subset(df, algorithm == 'CNN' & augmentation == 10))
  
  target_names = data.frame(id = c("CHEMBL1871_Ki","CHEMBL218_EC50","CHEMBL244_Ki","CHEMBL236_Ki","CHEMBL234_Ki","CHEMBL219_Ki","CHEMBL238_Ki","CHEMBL4203_Ki","CHEMBL2047_EC50","CHEMBL4616_EC50","CHEMBL2034_Ki","CHEMBL262_Ki","CHEMBL231_Ki","CHEMBL264_Ki","CHEMBL2835_Ki","CHEMBL2971_Ki","CHEMBL237_EC50","CHEMBL237_Ki","CHEMBL233_Ki","CHEMBL4792_Ki","CHEMBL239_EC50","CHEMBL3979_EC50","CHEMBL235_EC50","CHEMBL4005_Ki","CHEMBL2147_Ki","CHEMBL214_Ki","CHEMBL228_Ki","CHEMBL287_Ki","CHEMBL204_Ki","CHEMBL1862_Ki"),
                            name = c("AR","CB1","FX","DOR","D3R","D4R","DAT","CLK4","FXR","GHSR","GR","GSK3","HRH1","HRH3","JAK1","JAK2","KOR (a)","KOR (i)","MOR","OX2R","PPARa","PPARy","PPARd","PIK3CA","PIM1","5-HT1A","SERT","SOR","Thrombin","ABL1"))
  
  ## ML PCA loadings
  pca_all_ml = data_to_biplot(benchmark_ml, val_var="cliff_rmse" )
  bi_all_ml = pca_all_ml$bi
  bi_all_ml$algorithm = unlist(strsplit(as.character(bi_all_ml$name),  ' - '))[2*(1:length(bi_all_ml$name))-1 ]
  bi_all_ml$descriptor = unlist(strsplit(as.character(bi_all_ml$name),  ' - '))[2*(1:length(bi_all_ml$name)) ]
  bi_all_ml$algorithm[grepl('CHEMBL', bi_all_ml$name)] = ''
  bi_all_ml$descriptor[grepl('CHEMBL', bi_all_ml$name)] = ''
  bi_all_ml$col = "#27275d"
  bi_all_ml = subset(bi_all_ml, type == 'Loading')
  bi_all_ml$target_name = target_names$name[match(bi_all_ml$name, target_names$id)]
  
  # Make the actual plot
  pca_plot_ml = ggplot(bi_all_ml, aes(x = x, y =y)) +
    geom_hline(yintercept = 0, linetype = 'dashed', alpha = 0.25) +
    geom_vline(xintercept = 0, linetype = 'dashed', alpha = 0.25) +
    geom_point(aes(x, y ), colour = bi_all_ml$col, shape = 1,  size = 1, alpha = ifelse(bi_all_ml$type == 'Score', 0, 0.5)) +
    geom_point(aes(x, y ), colour = bi_all_ml$col, shape = 19,  size = 1, alpha = ifelse(bi_all_ml$type == 'Score', 0, 0.5)) +
    geom_text_repel(aes(label = target_name), colour = bi_all_ml$col, alpha = ifelse(bi_all_ml$type == 'Score', 0, 1), 
                    size = 2, segment.size = 0.25, force = 20, fontface="bold", max.iter = 1505, 
                    max.overlaps = 30, show.legend = FALSE)+
    labs(x = paste0('PC1 (',round(pca_all$scree$data$eig[1],1),'%)'),
         y = paste0('PC2 (',round(pca_all$scree$data$eig[2],1),'%)'),
         shape = 'Algorithm',  color = 'Descriptor') +
    guides(shape = guide_legend(override.aes = list(color  = "#27275d"), order = 2),
           color = guide_legend(override.aes = list(shape = 16), order = 1 )) +
    default_theme

  
  ## DL PCA loadings
  pca_all_dl = data_to_biplot(benchmark_dl, val_var="cliff_rmse" )
  bi_all_dl = pca_all_dl$bi
  bi_all_dl$algorithm = unlist(strsplit(as.character(bi_all_dl$name),  ' - '))[2*(1:length(bi_all_dl$name))-1 ]
  bi_all_dl$descriptor = unlist(strsplit(as.character(bi_all_dl$name),  ' - '))[2*(1:length(bi_all_dl$name)) ]
  bi_all_dl$algorithm[grepl('CHEMBL', bi_all_dl$name)] = ''
  bi_all_dl$descriptor[grepl('CHEMBL', bi_all_dl$name)] = ''
  
  bi_all_dl$col = "#27275d"
  
  # If 'Best' is on the left side of the plot, mirror everything. Makes it easer to compare with the previous plots
  if (subset(bi_all_dl, algorithm == 'Best')$x < 0){bi_all_dl$x = bi_all_dl$x*-1}
  if (subset(bi_all_dl, algorithm == 'Best')$y < 0){bi_all_dl$y = bi_all_dl$y*-1}
  
  bi_all_dl = subset(bi_all_dl, type == 'Loading')
  bi_all_dl$target_name = target_names$name[match(bi_all_dl$name, target_names$id)]
  
  # Make the actual plot
  pca_plot_dl = ggplot(bi_all_dl, aes(x = x, y =y)) +
    geom_hline(yintercept = 0, linetype = 'dashed', alpha = 0.25) +
    geom_vline(xintercept = 0, linetype = 'dashed', alpha = 0.25) +
    geom_point(aes(x, y ), colour = bi_all_dl$col, shape = 1,  size = 1, alpha = ifelse(bi_all_dl$type == 'Score', 0, 0.5)) +
    geom_point(aes(x, y ), colour = bi_all_dl$col, shape = 19,  size = 1, alpha = ifelse(bi_all_dl$type == 'Score', 0, 0.5)) +
    geom_text_repel(aes(label = target_name), colour = bi_all_dl$col, alpha = ifelse(bi_all_dl$type == 'Score', 0, 1), 
                    size = 2, segment.size = 0.25, force = 20, fontface="bold", max.iter = 1505, 
                    max.overlaps = 30, show.legend = FALSE)+
    labs(x = paste0('PC1 (',round(pca_all$scree$data$eig[1],1),'%)'),
         y = paste0('PC2 (',round(pca_all$scree$data$eig[2],1),'%)'),
         shape = 'Algorithm',  color = 'Descriptor') +
    guides(shape = guide_legend(override.aes = list(color  = "#27275d"), order = 2),
           color = guide_legend(override.aes = list(shape = 16), order = 1 )) +
    default_theme
  
  pca_plots = plot_grid(pca_plot_ml, pca_plot_dl, labels = c('a', 'b'), label_size=10, ncol=2, nrow=1, scale=1)
  
  return(pca_plots)
}



figure_s3 = function(df){
  
  # Select only the classical methods
  df = subset(df, algorithm %in% c('RF', 'SVM', 'GBM', 'KNN'))
  df$descriptor = factor(df$descriptor)
  df$algorithm = factor(df$algorithm)

  #### All subplots

  scatter_mini_plot = function(data, algo, descr, xlab='', ylab='', label=''){
    
    colours = c(rep('#b6b6b6', length(unique(df$algorithm))))
    colours[which(levels(df$algorithm) == algo)] = descr_cols$cols[which(descr_cols$descr == descr)]
    
    ggplot(data, aes(x=rmse, y=cliff_rmse, colour=algorithm))+
      geom_point(alpha = ifelse(data$algorithm == algo & data$descriptor == descr, 0, 0.2), shape=19, size=1, color='#b6b6b6' )+
      geom_point(alpha = ifelse(data$algorithm == algo & data$descriptor == descr, 0.5, 0), shape=1, size=1)+
      geom_point(alpha = ifelse(data$algorithm == algo & data$descriptor == descr, 0.5, 0), shape=19, size=1)+
      geom_abline(slope=1, intercept = 0, linetype='dashed', alpha=0.75)+
      geom_abline(slope=1, intercept = 0.25, linetype='dashed', alpha=0.25)+
      geom_abline(slope=1, intercept = -0.25, linetype='dashed', alpha=0.25)+
      geom_text(x=0.5, y=1.5, label=label, color = 'black', size=2.5, fontface="bold", hjust=0) +
      scale_x_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01))) +
      scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01))) +
      labs(x = xlab,  y = ylab) +
      coord_cartesian(ylim=c(0.25, 1.75), xlim=c(0.25, 1.75)) +
      scale_color_manual(values = colours)+
      default_theme +
      theme(legend.position = 'none')
  }
  
  whim_1 = scatter_mini_plot(df, 'KNN', 'WHIM', xlab='', ylab=bquote("RMSE"[cliff]), label='KNN')
  whim_2 = scatter_mini_plot(df, 'GBM', 'WHIM', xlab='', ylab='', label='GBM')
  whim_3 = scatter_mini_plot(df, 'RF', 'WHIM', xlab='', ylab='', label='RF')
  whim_4 = scatter_mini_plot(df, 'SVM', 'WHIM', xlab='', ylab='', label='SVM')
  
  phys_1 = scatter_mini_plot(df, 'KNN', 'Physchem', xlab='', ylab=bquote("RMSE"[cliff]), label='KNN')
  phys_2 = scatter_mini_plot(df, 'GBM', 'Physchem', xlab='', ylab='', label='GBM')
  phys_3 = scatter_mini_plot(df, 'RF', 'Physchem', xlab='', ylab='', label='RF')
  phys_4 = scatter_mini_plot(df, 'SVM', 'Physchem', xlab='', ylab='', label='SVM')
  
  maccs_1 = scatter_mini_plot(df, 'KNN', 'MACCs', xlab='', ylab=bquote("RMSE"[cliff]), label='KNN')
  maccs_2 = scatter_mini_plot(df, 'GBM', 'MACCs', xlab='', ylab='', label='GBM')
  maccs_3 = scatter_mini_plot(df, 'RF', 'MACCs', xlab='', ylab='', label='RF')
  maccs_4 = scatter_mini_plot(df, 'SVM', 'MACCs', xlab='', ylab='', label='SVM')
  
  ecfp_1 = scatter_mini_plot(df, 'KNN', 'ECFP', xlab='RMSE', ylab=bquote("RMSE"[cliff]), label='KNN')
  ecfp_2 = scatter_mini_plot(df, 'GBM', 'ECFP', xlab='RMSE', ylab='', label='GBM')
  ecfp_3 = scatter_mini_plot(df, 'RF', 'ECFP', xlab='RMSE', ylab='', label='RF')
  ecfp_4 = scatter_mini_plot(df, 'SVM', 'ECFP', xlab='RMSE', ylab='', label='SVM')
  
  scatters = plot_grid(whim_1, whim_2, whim_3, whim_4,
                       phys_1, phys_2, phys_3, phys_4,
                       maccs_1, maccs_2, maccs_3, maccs_4,
                       ecfp_1, ecfp_2, ecfp_3, ecfp_4,
                       ncol=4, nrow =4, scale = 1)
  
}



figure_s4 = function(df){
  df$method = paste0(df$algorithm, ' + ', df$descriptor)

  df$method[df$method == 'LSTM + SMILES'] = 'LSTM'
  df$method[df$method == 'CNN + SMILES'] = 'CNN'
  df$method[df$method == 'Transformer + SMILES'] = 'Transformer'
  df$method[df$method == 'AFP + Graph'] = 'AFP'
  df$method[df$method == 'GCN + Graph'] = 'GCN'
  df$method[df$method == 'MPNN + Graph'] = 'MPNN'
  df$method[df$method == 'GAT + Graph'] = 'GAT'

  kw = kruskal.test(cliff_rmse ~ method, data = df)
  print(kw)

  method_order = factor(df$method, levels = c("MLP + ECFP", "RF + ECFP", "GBM + ECFP", "SVM + ECFP", "KNN + ECFP", "RF + MACCs","GBM + MACCs","SVM + MACCs",
                                              "KNN + MACCs", "RF + Physchem", "GBM + Physchem", "SVM + Physchem", "KNN + Physchem",
                                              "RF + WHIM", "GBM + WHIM", "SVM + WHIM", "KNN + WHIM", "CNN", "LSTM", "Transformer",
                                              "AFP", "GCN", "MPNN", "GAT"))

  # significance plot
  wc = pairwise.wilcox.test(benchmark$cliff_rmse, method_order, p.adjust.method='BH')
  p = corrplot(wc$p.value, p.mat = wc$p.value, is.corr=F, col=c("#FFFFFF", rep("#D9D9D9", 19)),
           type = 'lower', tl.col = 'black', sig.level = c(0.05), method = 'shade', cl.pos = 'n',
           insig = 'label_sig', tl.srt = 45, col.lim = c(0, 1), pch.col = 'grey60', tl.cex = 0.3, pch.cex = 0.75, cl.cex = 0.3)

}


figure_s5 = function(df){

  df$descriptor = factor(df$descriptor, levels = c("Graph","ECFP","MACCs","Physchem","SMILES","WHIM"))
  colours = descr_cols$cols[match(levels(df$descriptor), descr_cols$descr)]

  datasets <- read_csv("MoleculeACE/Data/benchmark_data/metadata/datasets.csv")
  df$train_size = datasets$`Train compounds`[match(df$dataset, datasets$Dataset)]

  rmse_per_dataset <- group_by(df, dataset)
  rmse_per_dataset = summarise(rmse_per_dataset, mean_cliff_rmse=mean(cliff_rmse), min_cliff_rmse=min(cliff_rmse),
                               max_cliff_rmse=max(cliff_rmse), train_size=mean(train_size))

  p_datasize = ggplot(rmse_per_dataset, aes(y = mean_cliff_rmse, x = train_size))+
    geom_errorbar(aes(ymin=min_cliff_rmse, ymax=max_cliff_rmse),
                  colour='#27275d', width=0, size=0.5, alpha=0.5)+
    geom_point(size=1, shape=1, alpha=0.75, colour='#27275d') +
    geom_point(size=1, shape=19, alpha=0.75, colour='#27275d') +
    labs(x="Train molecules", y=bquote("RMSE"[cliff])) +
    scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.01,1.75), expand = expansion(mult = c(0.01, 0.01)))+
    scale_x_continuous(breaks = seq(500,3000,500), limits = c(450,3000), expand = expansion(mult = c(0.01, 0.01))) +
    coord_cartesian(ylim=c(0.25, 1.75))+
    scale_color_manual(values = colours)+
    guides(fill = 'none', color = 'none')+
    default_theme

  return(p_datasize)
}


figure_s6 = function(df){

  df$descriptor = factor(df$descriptor, levels = c("WHIM","Graph","Physchem","SMILES","MACCs","ECFP"))
  colours = descr_cols$cols[match(levels(df$descriptor), descr_cols$descr)]
  
  datasets <- read_csv("MoleculeACE/Data/benchmark_data/metadata/datasets.csv")
  df$receptor = datasets$`Receptor Class`[match(df$dataset, datasets$Dataset)]
  df$receptor[df$receptor %in% c('Protease', 'Transferase')] = 'Other'
  
  df$receptor[df$receptor == 'GPCR'] = 'GPCR (12)'
  df$receptor[df$receptor == 'Kinase'] = 'Kinase (6)'
  df$receptor[df$receptor == 'NR'] = 'NR (6)'
  df$receptor[df$receptor == 'Other'] = 'Other (6)'
  
  p = ggplot(df, aes(x=receptor, y=cliff_rmse, fill = descriptor))+
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=1, alpha=0.5) +
    geom_jitter(aes(color=descriptor), position=position_jitterdodge(0), size=1, shape=19, alpha=0.5) +
    geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.5,
                 outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=0.75) +
    scale_y_continuous(breaks = seq(0.25,1.75,0.25), limits = c(0.25,1.75), expand = expansion(mult = c(0.01, 0.01)))+
    scale_color_manual(values = colours)+
    scale_fill_manual(values = colours)+
    labs(x='Drug target type', y=bquote("RMSE"[cliff]), fill = 'Descriptor')+
    guides(fill = 'none')+
    default_theme +
    theme(legend.position = 'right')
  
  return(p)
  
}


print(figure_s1(benchmark))
dev.print(pdf, 'Experiments/figures/sup_fig_1.pdf', width = 7.205, height = 2)

print(figure_s2(benchmark))
dev.print(pdf, 'Experiments/figures/sup_fig_2.pdf', width = 7.205, height = 2)

print(figure_s3(benchmark))
dev.print(pdf, 'Experiments/figures/sup_fig_3.pdf', width = 7.205, height = 7.205)

print(figure_s4(benchmark))
dev.print(pdf, 'Experiments/figures/sup_fig_4.pdf', width = 3.504, height = 3.504)

print(figure_s5(benchmark))
dev.print(pdf, 'Experiments/figures/sup_fig_5.pdf', width = 3.504/2, height = 3.504/2)

print(figure_s6(benchmark))
dev.print(pdf, 'Experiments/figures/sup_fig_6.pdf', width = 7.205, height = 2)
