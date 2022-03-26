###### Set working dir and source some functions #######


setwd("/home/dwvtilborg/Dropbox/PycharmProjects/Activity_cliffs/")
source('Experiments/figures/R_scripts/vis_utils.R')


# Import data
benchmark = read_csv('MoleculeACE/Data/results/Benchmark_results.csv')
benchmark_npt = read_csv('MoleculeACE/Data/results/Benchmark_results_npt.csv')

benchmark$data_aug = 'Tranfer\nlearning'
benchmark_npt$data_aug = 'ab initio'
benchmark = rbind(benchmark, benchmark_npt)

# Rename some stuff
names(benchmark) = gsub('ac-rmse_soft_consensus','cliff_rmse',names(benchmark))

# Select the DL methods + a good and bad classical method
benchmark = subset(benchmark, algorithm == 'LSTM')

lstm_pretrain = ggplot(benchmark, aes(y = cliff_rmse, x = data_aug, group=dataset))+
    geom_point(size = 1, shape=1, alpha=0.5, color='#3BBF77') +
    geom_point(size = 1, shape=19, alpha=0.5, color='#3BBF77') +
    geom_line(alpha=0.1, size = 0.5, color='#3BBF77') +
    geom_boxplot(alpha=0.1, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
                 outlier.shape=NA, varwidth = FALSE, lwd=0.6, fatten=1, aes(group=data_aug), fill='#3BBF77') +
    labs(x='', y=bquote("RMSE"[cliff])) +
    scale_y_continuous(breaks = seq(0,5,0.5), expand = expansion(mult = c(0.01, 0.01))) +
    coord_cartesian(ylim=c(0, 2))+
    default_theme

print(lstm_pretrain)

dev.print(pdf, 'Experiments/figures/sup_fig_4.pdf', width = 1.752, height = 1.752)





