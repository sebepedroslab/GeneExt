args = commandArgs(trailingOnly = TRUE)
if(length(args)<4){
    message('Rscript script.r genic_peaks non_overlapping_peaks outfile peak_perc')
    quit()
}else{
    genic = args[1]
    noov = args[2]
    outfile = args[3]
    peak_perc = args[4]

    genic = read.table(genic)
    noov = read.table(noov)
    peak_perc = as.numeric(peak_perc)
    
    genic_cov = genic[,7]
    noov_cov = noov[,7]

    pdf(outfile)
    plot(density(log10(genic_cov)),col = 'red',main = 'Peak coverage',sub = paste0('peak perc: ',peak_perc),xlab = 'log10 Normalized peak coverage')
    lines(density(log10(noov_cov)),col = 'blue')
    thr = quantile(genic_cov,peak_perc/100)
    abline(v = log10(thr), lty = 2)
    dev.off()
}
