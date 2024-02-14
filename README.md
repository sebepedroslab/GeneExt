!['header'](./img/logo.png)

GeneExt takes as input scRNA-seq mapped reads and a gene annotation file (GTF or GFF, any version) and outputs an extended gene annotation file for improved scRNA-seq transcript quantification.

# Installation  
> **Note:** Users lacking a Conda installation are recommended to install [Miniforge](https://github.com/conda-forge/miniforge#miniforge).

Tool dependencies can be installed with `conda` or `mamba`:

```bash
# create environment
conda env create -n geneext -f environment.yaml

# activate environment
conda activate geneext
```

# Test run
Once dependencies are installed, try running `GeneExt` with sample data:

```bash
python geneext.py -g test_data/annotation.gtf -b test_data/alignments.bam -o result.gtf --peak_perc 0
```
(Note: `--peak_perc 0` is set to 0 to disable peak filtering as the test dataset is too small).
Please, first test whether the tool works properly on the test data! 

For details on how to obtain alignment file, please refer to the [manual](Manual.md).   
If problems persist, don't hesitate to contact the authors.

