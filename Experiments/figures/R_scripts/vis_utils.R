# Collection of functions for plotting figures

library(readr)
library(ggplot2)
library(cowplot)
library(ggsci)
library(plyr)
library(dplyr)
library(reshape2)
library(stringr)
library(ggrepel)
library(factoextra)
library(wesanderson)
library(scales)

# # Default colour palette I use for most plots
# cols = c('#121531', '#27275d' , '#2059a8', '#3bb1e6', '#89d7e9',
#          '#3c1347', '#6b238e' , '#8b0088','#da44a0', '#ee7ca4',
#          '#b7192a', '#db2823', '#fb5a00', '#ff9d00', '#fbd911',
#          '#13714c', '#269932', '#4bb54b', '#8dc409', '#c7da24')
# show_col(cols, ncol=5)

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

# Colours per descriptor type
descr_cols = list(cols = c('#2059a8','#3bb1e6','#fb5a00','#ff9d00','#ffffff', '#0B7344', '#40B87E',
                           '#27275d', '#27275d'),
                  descr =  c('WHIM', 'Physchem', 'MACCs', 'ECFP', "", 'Graph', 'SMILES',
                             'Best', 'Worst'))

data_to_bi_pca = function(df, axes = c(1, 2), loading_scaling=0.7, scale=T){
  #' Create a dataframe of eigenvectors and loadings from some data
  
  pca <- prcomp(df, scale = scale)
  
  scree = fviz_eig(pca)
  var <- facto_summarize(pca, element = "var", 
                         result = c("coord", "contrib", 'cos2'), 
                         axes = axes)
  ind <- facto_summarize(pca, element = "ind", 
                         result = c("coord", "contrib", 'cos2'), 
                         axes = axes)
  
  colnames(var)[2:3] <- c("x", "y")
  colnames(ind)[2:3] <- c("x", "y")
  
  # Scale Loadings 
  r <- min((max(ind[, "x"]) - min(ind[, "x"])/(max(var[, "x"]) - 
                                                 min(var[, "x"]))), (max(ind[, "y"]) - min(ind[, "y"])/
                                                                       (max(var[, "y"]) - min(var[, "y"]))))
  var[, c("x", "y")] <- var[, c("x", "y")] * r * loading_scaling
  
  # Merge the indivduals (eigenvectors) and variables 
  # (loading rotations, now scaled)
  var$type = rep('Loading', nrow(var))
  ind$type = rep('Score', nrow(ind))
  bi = rbind(ind, var)
  bi$cos2[bi$type == 'Score'] = 0
  bi$contrib[bi$type == 'Score'] = 0
  
  return( list('bi'=bi, 'pca'=pca, 'scree'=scree) )
}


data_to_biplot = function(pca_dat, algo_col='algorithm', 
                          descr_col='descriptor', val_var="cliff_rmse", 
                          loading_scaling=1.6){
  #' convert a dataframe into pca data + best/worst scaling
  
  pca_dat$method = paste0(pca_dat[[algo_col]],'9', pca_dat[[descr_col]])
  M_all = dcast(data = pca_dat, formula = method~dataset, value.var = val_var)
  
  rownames(M_all) = str_to_sentence(gsub('_', ' ', unlist(M_all[1])))
  M_all = M_all[2:ncol(M_all)]
  M_all= data.frame(t(M_all))
  
  
  M_all$Best = apply(M_all, 1, FUN = min)
  M_all$Worst = apply(M_all, 1, FUN = max)
  colnames(M_all) = gsub('9', ' - ', gsub('\\.', ' ', colnames(M_all)))
  
  
  pca_all = data_to_bi_pca(t(M_all), loading_scaling=loading_scaling, scale=F)
  scree_all = pca_all$scree
  
  pca_all$bi$name = gsub('Best', 'Best - Best', pca_all$bi$name)
  pca_all$bi$name = gsub('Worst', 'Worst - Worst', pca_all$bi$name)
  
  return(pca_all)
  
}


# GGplot default theme I use
default_theme = theme(
  panel.border = element_rect(colour = "black", size = 1, fill = NA),
  panel.background = element_blank(),
  plot.title = element_text(hjust = 0.5, face = "plain"),
  # axis.ticks.y = element_blank(),
  axis.text.y = element_text(size=6, face="plain", colour = "black"),
  axis.text.x = element_text(size=6, face="plain", colour = "black"),
  axis.title.x = element_text(size=6, face="plain", colour = "black"),
  axis.title.y = element_text(size=6, face="plain", colour = "black"),
  legend.key = element_blank(),
  legend.position = 'bottom', legend.box = "vertical",
  legend.title = element_blank(),
  legend.background = element_blank(),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank())
