!['header'](./img/logo.png)

GeneExt takes as input scRNA-seq mapped reads and a gene annotation file (GTF or GFF, any version) and outputs an extended gene annotation file for improved scRNA-seq transcript quantification.

# Installation  

> **Note:** Users lacking a Conda installation are recommended to install [Miniforge](https://github.com/conda-forge/miniforge#miniforge).

Tool dependencies can be installed with `conda` or `mamba`:

```bash
# create environment
mamba env create -n geneext -f environment.yaml
mamba activate geneext 
```

Install `macs2` separately with `pip`:  
```bash
pip install macs2
```

# Test run
Once dependencies are installed, try running `GeneExt` with sample data:

```bash
python geneext.py -g test_data/annotation.gtf -b test_data/alignments.bam -o result.gtf --force --orphan
```


This should generate `result.gtf` file and a temporary directory `tmp/` with the intermediate files useful for debugging. 
For instance:  
	- `tmp/extensions.tsv` - gene_id, peak_id, extension length - usefull for plotting the distribution of extensions 
	- `tmp/allpeaks_coverage.bed` - all peaks with normalized coverage 
	- `tmp/allpeaks_noov_fcov.bed` - all peaks not overlapping the genes that have been coverage-filtered 
	- `tmp/orphan_merged.bed` - the final list of the orphan peaks added 





The resulting gtf file will contain:  
	- input features - untouched   
	- input transcripts extended - the 2nd column (source) changed to 'GeneExt'   
	- inferred orphan peaks - exon,transcript/gene triplets per orphan cluster; 'GeneExt_orphan'  

The updated features can be easily tracked by their source column (2nd):  

```bash
cat result.gtf | awk '$3=="gene"' | cut -f 2 | sort | uniq -c 
#     35 Genbank
#     14 GeneExt_orphan
cat result.gtf | awk '$3=="transcript"' | cut -f 2 | sort | uniq -c 
#     14 Genbank
#     21 GeneExt
#     14 GeneExt_orphan
```
The output above suggests there are 14 orphan peaks (`GeneExt_orphan`), and 21 genes extended (the source of the transcript has changed to `GeneExt`); 14 input genes have been left unchanged. 

# Notes

Most errors with `GeneExt` come from improperly formatted files. If you encounter errors, please, try standardizing your annotation file with [`AGAT`](https://agat.readthedocs.io/en/latest/tools/agat_convert_sp_gxf2gxf.html). 


For details on how to obtain the alignment file, please refer to the [manual](Manual.md).   
If problems persist, don't hesitate to contact the authors.



## Citation

If you use this tool, please cite:

> Zolotarov, G., Grau-Bové, X., & Sebé-Pedrós, A.  
> **GeneExt: a gene model extension tool for enhanced single-cell RNA-seq analysis**  
> *bioRxiv* 2023.12.05.570120  
> https://doi.org/10.1101/2023.12.05.570120
