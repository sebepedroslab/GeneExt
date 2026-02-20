!['header'](./img/logo.png)  
# Manual  

Genomes often have incomplete annotations of their 3-prime untranslated regions (3'-UTRs). At the same time, some of the most popular single-cell RNA sequencing methods are biased towards 3' ends of mRNA molecules. In result, this creates a bias in gene counting for genes with missing fragments of 3'-UTRs:   
![Gene_counting](./img/gene_counting_problem.png)

`GeneExt` aims to refine gene models in the reference genome by leveraging the scRNA-seq data itself (or similar 3'-biased transcriptomics data). `GeneExt` not only improves UMI count accuracy per gene but also resolves gene overlap issues.

## How it works  

![Pipeline](./img/pipeline.png)

1. `GeneExt` accepts alignment files from any 3'-end biased single-cell (or bulk RNA-seq) protocol. It will then call peaks from this data using `macs2` software. 
2. For every gene, the most downstream peak will be chosen as a new mRNA cleavage site. The maximal distance from a gene to a peak is controlled by an `-m` parameter (see below).  
3. After genes are extended, `GeneExt` will write an output file which can be used to build a genome reference (e.g. with `cellranger mkref`).  
4. If the `--orphan` option is used, `GeneExt` will use the intergenic peaks not previously assigned to an upstream gene (only those surviving the coverage filter step) to add putative new genes to the final annotation. 

## Important parameters   

### --m Maximal extension length   

`-m` parameter specifies the maximum distance the gene is allowed to be extended for. Setting `-m` to larger values will almost always result in longer extensions of genes and thus more reads counted per gene. However, the genome annotation is guaranteed to be __missing some genes__. In such cases, you may actually __misassign the reads to the gene they don't belong to__.

!['Gene extension'](./img/max_ext.png)

Thus, instead of setting `-m` to unrealistically big values, we advice setting it to something biologically meaningful (e.g 1x-2x of median length of a gene) and to use it along with calling "orphan peaks" ( `--orphan` option, vis [Orphan peaks](#orphan-peaks)).


### --orphan Extension modes  

Use this option to keep [Orphan peaks](#orphan-peaks).


### --peak_perc Filtering peaks based on coverage  

To make `GeneExt` more conservative in peak calling, peaks are filtered based on the average coverage.   
After calling the peaks, `GeneExt` will calculate per-base coverage distribution for __genic__ peaks (i.e. peaks overlapping genes). This distribution is then used to filter __intergenic peaks__. `--peakp` sets a quantile of that distribution above which intergenic peaks are retained.  

![Peak filtering](./img/peak_filtering.png)   

Thus, decreasing `--peakp` will result in more peaks called and _vice versa_.   

## --clip_5prime Gene overlap clipping  

`--clip_5prime` option enables `GeneExt` to resolve 5'-overlaps between genes.   
Depending on the behavior of the UMI demultiplexing software used, gene overlaps can cause:
1. the upstream gene to not be quantified (if 3′ biased scRNA-seq reads mapped into the overlapping region are discarded such as in cellranger)  
2. the downstream gene to have two distinct confounding expression signals (if reads are assigned to both).      


`GeneExt` will clip the downstream gene giving the preference to an upstream one:  

![Peak filtering](./img/5clip.png)   

> [!Warning]
> This clipping procedure is done without respecting protein-coding information containted in the downstream gene. Use with caution!   


## Input & Output   
To run `GeneExt`, you will need the following:  

1. scRNA-seq dataset mapped to the genome - `.bam` alignment file (vis ["Where do I get a .bam file?"](#how-to-get-a-bam-file))  
2. Genome annotation in the `gff` or `gtf` format  

`GeneExt` accepts following annotion input formats and converts them to the output:   
* `gtf` &rarr; `gtf`  
* `gff` &rarr; `gff`  

In general, `GeneExt` will try to output a properly formatted `gtf` file that can be used as an input to `cellranger mkref`. However, since `gtf` files vary in their attributes, this may not always be possible (vis [Input troubleshooting](#input-troubleshooting)).
Please, ensure your genome annotation file is properly formatted!   

## How to get a .bam file?   

If you already have used `cellranger`, then you can simply use its `.bam` file (`possorted_genome.bam`). Alternatively, you may generate an alignment yourself with any splice-aware aligner. 

> [!Warning]
> For now, `GeneExt` only accepths a single alignment file, so if you have multiple sequencing datasets, you should concatenate your scRNA-seq fastq file for the following step: 

```cat lane1.R2.fastq.gz lane2.R2.fastq.gz > data/cells.R2.fastq.gz```

Below is an example of how to generate such an alignment with `STAR` aligner:  

```bash
STARIDX=~/genomes/star_idx/ # path to the STAR index
GENOMEFA=~/genomes/genome.fa # path to genome sequence fasta file for genome index 

R2=data/cells.R2.fastq.gz # R2 reads from a 10x experiment 
NCPU=5 # number of cores to use for mapping 

# generate the genome index (skip if you already have a genome index)   
STAR --runMode genomeGenerate --runThreadN $NCPU --genomeDir $STARIDX --genomeFastaFiles $GENOMEFA

# run STAR alignment
STAR --genomeDir $STARIDX --outFilterMultimapNmax 10 --runThreadN $NCPU --readFilesIn $R2 --outFileNamePrefix cells --readFilesCommand zcat --outSAMtype BAM SortedByCoordinate --outSAMattributes Standard
```

Resulting `cellsAligned.sortedByCoord.out.bam` can be used as an input for the `GeneExt`.  

## BAM subsampling   

Sometimes, the dataset may be too large for `GeneExt` to run in meaninful time, you can subsample the `.bam` file to N reads using an option `--subsamplebam N`. This will significantly speed up the pipeline, but may come at a cost of missing some peaks.  

## Orphan peaks  

### Definition  

The majority peaks not be assigned to any gene due to the distance (`-m` parameter). However, some of these peaks will correspond to really long 3'-UTRs or __unannotated genes__. 

To capture cases like this, `GeneExt` provides an option to keep the peaks that pass coverage filtering but haven't been assigned to any gene (e.g. they are located too far from any genic region) with an `--orphan` option.  

__Note:__ it may happen that you will get a lot of "orphan" peaks in your annotation file  (e.g. 100 000). Don't worry, having these peaks in your genome annotation will not affect the counting. After you obtain a count matrix, you can always filter these peaks in the downstream analyses.    

### Orphan peak merging      

Missing genes may be represented by multiple orphan peaks corresponding to exonic regions. Having such peaks will lead to including highly correlated features which is undesirable for single-cell RNAseq analyses. 
By default, `GeneExt` will try to merge such peaks by distance unless `--nomerge` is specified.  

![Orphan_merging](./img/peak_clustering.png)

Default settings are the following:  
* Maximum distance between the peaks (`--orphan_maxdist`) - 75-th percentile of intron sizes.  
* Maximum size of the orphan peak cluster (`--orphan_maxsize`) - median gene length.  

The merged peaks are represented by a single continuos region.  

## Input troubleshooting  

> "Properly formatted .gtf files are all alike; each bad .gtf is broken in its own way."
>
> Anna Karenina principle for genome annotation files

By far the most probable reason for `GeneExt` to fail are the problems with input annotation file. Below is a description of a minimal annotation file `GeneExt` can work with.  

Imagine a gene with a single transcript and 2 exons. A properly formatted minimal GFF file should look like the following:   
```
chr1  source  gene 1 100 .  + . ID=gene1;
chr1  source  transcript  1 100 . + . ID=transcript1;Parent=gene1
chr1  source  exon  1 40  . + . ID=exon1;Parent=transcript1
chr1  source  exon  70 100  . + . ID=exon1;Parent=transcript1    
```
GTF file is similar, but the 9-th column contains `gene_id` and  `transcript_id` attributes:   
```
chr1  source  gene 1 100 .  + . gene_id "gene1";
chr1  source  transcript  1 100 . + . gene_id "gene1"; transcript_id "transcript1"
chr1  source  exon  1 40  . + . gene_id "gene1"; transcript_id "transcript1"
chr1  source  exon  70 100  . + . gene_id "gene1"; transcript_id "transcript1"   
```

The most common problems with gtf/gff files:   
1. Missing "gene" features - `GeneExt` will try to infer missing "gene" features.
2. Missing "transcript" features - `GeneExt` can't infer transcripts at the moment. Please, check if your annotation file at least contains exons and transcripts.    
3. "gene" and "transcript" features have the same IDs - `GeneExt` may have troubles when parsing the file.  

> [!Warning]
> Please, make sure your GTF file at least has "transcript" and "exon" features. Transcripts should __NOT__ have the same IDs as genes even if there is a single transcript per gene!  


# FAQs  

## GeneExt does not accept my annotation file  

See [Input troubleshooting](#input-troubleshooting)
## GeneExt takes a long time to run. How can I speed it up?  

1. `.bam` subsampling:    
  By default, `GeneExt` will use the whole dataset to call the peaks. This may be computationally costly for big datasets (>50M reads). You can use `--subsamplebam 10000000` to randomly sample 10M reads (or any other amount). Keep in mind, __using more data is always better.__  
2. Use multiple threads with `-j` option.
2. Split your bam files by chromosomes:  
  You can split your genome into individual chromosomes / contigs and run `GeneExt` on each of them separately  

## I get too many peaks. How should I filter them?   
The results of peak calling depend on the  dataset quality. By default, `GeneExt` filters the peaks based on the coverage __before__ gene extension.  
However, as is stated above, having many "orphan" peaks in your annotation __will not affect gene counting__ but may help preserving valuable information about cell type heterogeneity in the data.    
If you wish to remove more peaks, you may increase the `--orphan_maxdist` parameter. This will lead to more orphan peaks clustered to end up in the clusters of bigger size that will be removed (`--orphan_maxsize`).

## Some orphan peaks look like missing genes - how can I link them? 
For the specified peaks you want to merge, you can manually change the `gene.id` attribute in every peak to a common value (e.g. an 'unknown_gene_1'). If you observe a lot of such cases, you can try increasing parameters for orphan peak clustering and merging (`--orphan_maxdist`,`--orphan_maxsize`) so that these orphan peaks will end up called as clusters.   


