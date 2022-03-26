###### Set working dir and source some functions #######


setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")
source('Experiments/figures/R_scripts/vis_utils.R')


###### Data prep #######


# Import data
benchmark = read_csv('MoleculeACE/Data/results/Benchmark_results.csv')

names(benchmark) = gsub('ac-rmse_soft_consensus','cliff_rmse', names(benchmark))
benchmark$descriptor[benchmark$descriptor == 'Canonical'] = 'Graph'
benchmark$descriptor[benchmark$descriptor == 'Attentivefp'] = 'Graph'

# Select only the classical methods
benchmark_ml = subset(benchmark, algorithm %in% c('RF', 'SVM', 'GBM', 'KNN'))
# Select the DL methods + a good and bad classical method
benchmark_dl = rbind(subset(benchmark, algorithm %in% c('GCN', 'MPNN', 'AFP', "GAT", 'DNN')),
                     subset(benchmark, algorithm == 'SVM' & descriptor == 'ECFP'),
                     subset(benchmark, algorithm == 'LSTM' & augmentation == 10),
                     subset(benchmark, algorithm == 'CNN' & augmentation == 10))


target_names = data.frame(id = c("CHEMBL1871_Ki","CHEMBL218_EC50","CHEMBL244_Ki","CHEMBL236_Ki","CHEMBL234_Ki","CHEMBL219_Ki","CHEMBL238_Ki","CHEMBL4203_Ki","CHEMBL2047_EC50","CHEMBL4616_EC50","CHEMBL2034_Ki","CHEMBL262_Ki","CHEMBL231_Ki","CHEMBL264_Ki","CHEMBL2835_Ki","CHEMBL2971_Ki","CHEMBL237_EC50","CHEMBL237_Ki","CHEMBL233_Ki","CHEMBL4792_Ki","CHEMBL239_EC50","CHEMBL3979_EC50","CHEMBL235_EC50","CHEMBL4005_Ki","CHEMBL2147_Ki","CHEMBL214_Ki","CHEMBL228_Ki","CHEMBL287_Ki","CHEMBL204_Ki","CHEMBL1862_Ki"),
                          name = c("AR","CB1","FX","DOR","D3R","D4R","DAT","CLK4","FXR","GHSR","GR","GSK3","HRH1","HRH3","JAK1","JAK2","KOR (a)","KOR (i)","MOR","OX2R","PPARa","PPARy","PPARd","PIK3CA","PIM1","5-HT1A","SERT","SOR","Thrombin","ABL1"))

##### ML PCA loadings ######

pca_all_ml = data_to_biplot(benchmark_ml, val_var="cliff_rmse" )
bi_all_ml = pca_all_ml$bi

bi_all_ml$algorithm = unlist(strsplit(as.character(bi_all_ml$name),  ' - '))[2*(1:length(bi_all_ml$name))-1 ]
bi_all_ml$descriptor = unlist(strsplit(as.character(bi_all_ml$name),  ' - '))[2*(1:length(bi_all_ml$name)) ]
# 
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

print(pca_plot_ml)



#### DL PCA loadings ######

benchmark_dl = subset(benchmark_dl, algorithm != 'GAT')
pca_all_dl = data_to_biplot(benchmark_dl, val_var="cliff_rmse" )
bi_all_dl = pca_all_dl$bi

bi_all_dl$algorithm = unlist(strsplit(as.character(bi_all_dl$name),  ' - '))[2*(1:length(bi_all_dl$name))-1 ]
bi_all_dl$descriptor = unlist(strsplit(as.character(bi_all_dl$name),  ' - '))[2*(1:length(bi_all_dl$name)) ]
# 
bi_all_dl$algorithm[grepl('CHEMBL', bi_all_dl$name)] = ''
bi_all_dl$descriptor[grepl('CHEMBL', bi_all_dl$name)] = ''

bi_all_dl$col = "#27275d"

# If 'Best' is on the left side of the plot, mirror everything. Makes it easer to compare with the previous plots
if (subset(bi_all_dl, algorithm == 'Best')$x < 0){
  bi_all_dl$x = bi_all_dl$x*-1
}
if (subset(bi_all_dl, algorithm == 'Best')$y < 0){
  bi_all_dl$y = bi_all_dl$y*-1
}

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

print(pca_plot_dl)



##### Combine subplots ######

pca_plots = plot_grid(pca_plot_ml, pca_plot_dl, labels = c('a', 'b'), label_size=10, ncol=2, nrow=1, scale=1)
print(pca_plots)

dev.print(pdf, 'Experiments/figures/sup_fig_2.pdf', width = 7.205, height = 2.5)


