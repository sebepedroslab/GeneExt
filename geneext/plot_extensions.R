#infile = 'tmp/extensions.tsv'
#outfile = 'extensions.pdf'
args = commandArgs(trailingOnly = TRUE)
if(length(args)<2){
    message('Rscript script.r infile outfile')
    quit()
}else{
    infile = args[1]
    outfile = args[2]

    df = read.table(infile)
    pdf(outfile)
    plot(density(df[,3]),xlab = 'Gene extension, bp',main = 'Distribution of gene extensions',sub = paste0('Number of genes: ',nrow(df)))
    dev.off()
}

