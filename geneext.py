#!/usr/bin/env python3

import argparse
from argparse import RawTextHelpFormatter
from geneext.config import read_yaml_config
from geneext import helper
from geneext.helper import pipeline_error_print

from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape

import os
from os import path

import sys
import pandas as pd

parser = argparse.ArgumentParser(description=
"""
Program: GeneExt
Version: 1.0
""",
                           formatter_class=RawTextHelpFormatter)
parser.add_argument('-g','--genome', default= None,help = 'Genome .gtf/.gff/.bed file.' ,required = True) 
parser.add_argument('-b','--bam', default= None,help = 'Input .bam file.')
parser.add_argument('-p','--peaks', default= None,help = 'Peaks .bed file. Incompatible with -b.\nIf provided, extension is performed using specified coordinates in .bed format.\n(Can be seful in cases of FLAM-seq / Nano-3P-seq data or when manual filtering of the peaks is needed.)') 
parser.add_argument('-o','--output', default = None, help = 'Output annotation file.\n\n\n================ Additional arguments ================\n',required = False)
parser.add_argument('-m','--maxdist', default = None, help = 'Maximal distance for gene extension.\nIf not set, a median length of gene (genomic span!) is used.')

parser.add_argument('-inf', default = None, help = 'Input genes file format, if None, will be guessed from a file extension.')
parser.add_argument('-ouf', default = None, help = 'Output file format, if not given, will be guessed from a file extension.')
parser.add_argument('-t', default = None, help = 'Temporary directory. [tmp_{output_file_prefix}]')
parser.add_argument('-tag', default = str('GeneExt'), help = 'Tag to be added to the fake gene source and IDs so these can be easily identified downstream. [GE]')
parser.add_argument('-v','--verbose', default = int(0), help = 'Verbosity level. [0],1,2,3')
parser.add_argument('-j','--jobs', default = '1', help = 'Number of parallel cores. [1]')
parser.add_argument('--output_mode', default = 'new_transcript', help = 'How to extend the gene (only for .gff/.gtf files) [new_transcript]\n\t* new_transcript - creates a new transcript feature with the last exon extended\n\t* new_exon - creates an extended last exon')
parser.add_argument('-l','--longest',action='store_true', help = 'Whether to select the longest isoform per gene.')
parser.add_argument('--clip_strand',default = 'sense',help = 'How to treat gene extension overlaps.\nsense - default,restrict overlaps into downstream genes on the same strand\nboth - restrict overlaps regardless of the strand.')
parser.add_argument('--clip_5prime', action='store_true', help = 
'''Use this to clip overlaps between genes. The downstream gene will be clipped.
CAVE: use carefully if the genome contains many overlapping genes (e.g. mitochondrial genomes).

================ Orphan peaks ================
Orpan peaks are peaks that could not be assigned to any gene due to the distance.
Since those can correspond to missing 3prime UTRs or missing genes, we suggest keeping them.
Orphan peaks are merged into orphan peak clusters by distance and then are kept as if they were genes. 

''')
parser.add_argument('-orphan','--orphan',action='store_true', help = 'Whether to add orphan peaks')
parser.add_argument('-orphan_maxdist','--orphan_maxdist', default = None, help = 'Orphan peak merging: Maximum distance between orphan peaks to merge. [75-th percentile of intron length, bp]')
parser.add_argument('-orphan_maxsize','--orphan_maxsize', default = None, help = 'Orphan peak merging: Maximum size of an orphan peak cluster. Default: [median gene length, bp]')
#parser.add_argument('--mean_coverage', action='store_true', help = 'Whether to use mean coverage for peak filtering.\nMean coverage = [ # mapping reads]/[peak width].')
parser.add_argument('-peak_perc','--peak_perc',default = 25, help = 'Coverage threshold (percentile of macs2 genic peaks coverage). [1-99, 25 by default].\nAll peaks called with macs2 are required to have a coverage AT LEAST as N-th percentile of the peaks falling within genic regions.\nThis parameter allows to filter out the peaks based on the coverage BEFORE gene extension.')
parser.add_argument('-nocluster','--nocluster', action='store_true', help = 'Do not merge orphan peaks based on distance.\n\n\n================ Miscellaneous ================\n')

parser.add_argument('-subsamplebam','--subsamplebam',default = None, help = 'If set, will subsample bam to N reads before the peak calling. Useful for large datasets. Bam file should be indexed.\nDefault: None')
#parser.add_argument('--report', action='store_true', help = 'Use this option to generate a PDF report.')
parser.add_argument('-keep','--keep_intermediate_files', action='store_true', help = 'Use this to keep .bam and other temporary files in the a temporary directory. Useful for troubleshooting.')
#parser.add_argument('--estimate', action='store_true', help = 'Use this to just estimate intergenic read proportion.\nUseful for quick checking intergenic mapping rate.')
parser.add_argument('--onlyfix', action='store_true', help = 'If set, GeneExt will only try to fix the annotation, no extension is performed.')
parser.add_argument('-force','--force', action='store_true', help = 'If set, GeneExt will ignore previously computed files and will re-run everythng from scratch.')
parser.add_argument('-rerun','--rerun', action='store_true', help = 'If set, GeneExt will try to re-run the analysis using previously computed files it finds in the temporary directory (subsampled data, MACS2 results, peaks with coverage etc.).')

########### Pipeline Functions ################
###############################################
   

# Visual settings
#def pipeline_error_print(x=None):
#    console.print(x, style="bold red")
#    #print("\n"+x + '\n')
#    #os.system('cat %s/geneext/err.txt' % scriptloc)
#    quit()

def print_task(x):
    console.print('%s ...' % x,style = 'bold yellow', end = end)


def parse_input_format():
    if args.inf is not None:
        infmt = args.inf
        if verbose > 0:
            print('Input: %s; Specified format: %s'  % (genefile,infmt))
    else:
        # guess from the output file name 
        infmt = helper.guess_format_fromfile(genefile)
        if verbose > 0:
            print('Input: %s, guessed format: %s' % (genefile,infmt))
    if not infmt in ['bed','gff','gtf']:
        pipeline_error_print('Unknown input format "%s"!\nPlease either make sure the format is written properly or specify it yourself using --ouf option.' % infmt)
    return(infmt)
def parse_output_format():
    if args.ouf is not None:
        outfmt = args.ouf
        if verbose > 0:
            print('Output: %s; Specified format: %s' % (outputfile,outfmt))
    else:
        # guess an output format from the name extension
        if helper.get_extension(outputfile)!= "":
            outfmt = helper.get_extension(outputfile)
        else:
            # use input file format
            outfmt = infmt
            if verbose > 0:
                console.print('Output format can not be guessed from output file name -> using input file format %s' % infmt,style = 'bold yellow')
        if verbose > 0:
            print('Output: %s, guessed format: %s' % (outputfile,outfmt))
    if not outfmt in ['bed','gff','gtf']:
        pipeline_error_print('Unknown output format "%s"!\nPlease either make sure the format is written properly or specify it yourself using --ouf option.' % outfmt)
    return(outfmt)      

#def compare_infmt_outfmt(infmt,outfmt):
#    if infmt != outfmt:
#        pipeline_error_print('Please, make sure the input and the output have the same format!')


def run_peakcalling():
    helper.split_strands(bamfile,tempdir,verbose = verbose,threads = threads)
    helper.run_macs2(tempdir+'/' + 'plus.bam','plus',tempdir,verbose = verbose)
    helper.run_macs2(tempdir+'/' + 'minus.bam','minus',tempdir,verbose = verbose)
    helper.collect_macs_beds(outdir = tempdir,outfile = rawpeaks,verbose = verbose)

def get_orphan(genefile = None,genefile_ext_bed = None,peaks_bed = None,orphan_bed = None,infmt = None, outfmt = None,verbose = False,merge = False):
        """Store peaks not falling within new genic regions as orphan peaks """
        if infmt != 'bed':
            # remove orphan peaks overlapping extended regions:  
            helper.gxf2bed(infile = genefile,outfile = genefile_ext_bed,featuretype = 'gene')
            helper.outersect(inputbed_a = peaks_bed,inputbed_b=genefile_ext_bed,outputbed = orphan_bed,by_strand = False,verbose = verbose,f = 0.0001)
        else:
            pipeline_error_print("Don't know how to add orphan peaks!")


# Pipeline: orphan peaks 
def run_orphan():
    print_task('Adding orphan peaks')
    genefile_ext_bed = tempdir + '/' + 'genes_ext.bed'
    orphan_bed = tempdir + '/' + 'orphan.bed'
    # if no bam file provided - no coverage filtering - all peaks not overlapping the genes. Else - coverage-filtered peaks. 
    get_orphan(genefile = outputfile, genefile_ext_bed= genefile_ext_bed,peaks_bed = peaksfilt,orphan_bed = orphan_bed,infmt = infmt, outfmt = outfmt, verbose = verbose)
    console.print('done',style = 'bold green') 
    if verbose:
        print(f'Orphan peaks: {helper.count_lines(orphan_bed)} orphan peaks generated')
    if do_orphan_merge:
        if verbose:
            print('Orphan peaks: merging by distance.')
        orphan_merged_bed  = tempdir + '/' + 'orphan_merged.bed'
        chrnamesfile = tempdir + '/chr_names.txt'
        helper.get_chr_names(bamfile, chrnamesfile,verbose = verbose)
        helper.merge_orphan_distance(orphan_bed = orphan_bed,chr_names = chrnamesfile,orphan_merged_bed = orphan_merged_bed,genic_bed = genefile_ext_bed,tempdir = tempdir,maxdist = orphan_maximum_distance,maxsize = orphan_maximum_size, verbose = verbose)
        N_init = helper.count_lines(orphan_bed)
        N_merge = helper.count_lines(orphan_merged_bed)
        console.print(f'Merged {N_init} orphan peaks into {N_merge} clusters:',style = 'bold yellow',end = '')
        console.print(f' {orphan_merged_bed}',style = 'yellow')
        if verbose:
            print('Orphan peaks: merged peaks - %s' % orphan_merged_bed)
        helper.add_orphan_peaks(infile = outputfile,peaksbed=orphan_merged_bed,fmt = outfmt,verbose=verbose,tag = tag)
        orphan_bed = orphan_merged_bed
    else:
        if verbose>2:
            print('Orphan peaks: no merging -> directly adding orphan peaks.')
        helper.add_orphan_peaks(infile = outputfile,peaksbed=orphan_bed,fmt = outfmt,verbose=verbose,tag = tag)  
    console.print('Added %s orphan peaks / clusters.' % helper.count_lines(orphan_bed),style = 'bold yellow')

# Reporting functions 

def report_extensions(file_path,orphan_bed = None,n_genes = None):
    def is_file_empty(file_path):
        """Check if a file has zero lines."""
        with open(file_path, 'r') as file:
            return not any(file)
    """Load a tabular file, report the number of rows and median value of the last column."""
    if not is_file_empty(file_path):
        df = pd.read_csv(file_path,sep = '\t')
        num_rows = len(df)
        median_value = round(df.iloc[:, -1].median(), 1)
        console.print('Extended %s/%s genes\nMedian extension length: %s bp' % (num_rows,n_genes,median_value),style = 'bold green')
        #if orphan_bed:
        #    # report the number of orphan peaks / clusters added 
        #    df = pd.read_csv(orphan_bed,sep = '\t')
        #    num_rows = len(df)
        #    console.print('Added %s orphan peaks / clusters.' % num_rows,style = 'bold green')
            
    else:
        console.print('No genes could be extended',style = 'bold red')


def clean_tmp(tempdir = None):
    # clean temporary directory of big files 
    toremove = [tempdir+'/'+x for x in os.listdir(tempdir) if '.bam' in x or  x[0] == '_']
    for file in toremove:
        if verbose > 2:
            print("Removing %s" % file)
        os.remove(file)

def estimate_mapping(tempdir = None,bamfile = None,genefile = None,infmt = None,threads = 1, verbose = False,orphanbed = None):
        # TODO: add orphan peak estimate in case there are orphan peaks? 
        # if orphan peaks present, estimate maping in the orphan peaks as well. 
        genicbed = tempdir + '/genic.bed'
        intergenicbed = tempdir + '/intergenic.bed'
        chrsizesfile = tempdir + '/chr_sizes.tab'

        # prepare the files
        helper.get_chr_sizes(tempdir = tempdir, bamfile = bamfile, outfile = chrsizesfile, verbose = verbose)
        helper.get_genic_beds(genomeanno=genefile,genomechr=chrsizesfile,verbose = verbose,infmt = infmt,genicbed=genicbed,intergenicbed=intergenicbed)

        Ntot = helper.count_reads(bamfile=bamfile,bed = None,flags = '',threads=threads,verbose = verbose)
        Nmap = helper.count_reads(bamfile=bamfile,bed = None,flags = '-F 4',threads=threads,verbose = verbose)
        Ngen = helper.count_reads(bamfile=bamfile,bed = genicbed,flags = '',threads=threads,verbose = verbose)
        Nigen = Nmap - Ngen

        if orphanbed:
            # compute for genic regions
            genicnoorphanbed = tempdir + '/genic_no_orphan.bed'
            helper.outersect(inputbed_a=genicbed,inputbed_b=orphanbed,outputbed = genicnoorphanbed,verbose = verbose)
            Ngen = helper.count_reads(bamfile=bamfile,bed = genicnoorphanbed,flags = '',threads=1,verbose = verbose)
            Norph = helper.count_reads(bamfile=bamfile,bed = orphanbed,flags = '',threads=1,verbose = verbose)
        else:
            Norph = 0
        return(Ntot,Nmap,Ngen,Nigen,Norph)

# reporting mapping stats
def report_stats(filename=None,Ntot=None,Nmap=None,Ngen=None,Nigen=None,Norph=None):
    """Report mapping statistics in a way similar to cellranger"""
    o = '%s:\nTotal reads: %s\nMapped reads: %s (total: %s %%)\nGenic reads: %s (total: %s %%; mapped: %s %%)\nOrphan peaks: %s (total: %s %%; mapped: %s %%)\nIntergenic reads: %s (total: %s %%; mapped: %s %%)' % (filename,str(Ntot),str(Nmap),str(round(Nmap/Ntot*100,2)),str(Ngen),str(round(Ngen/Ntot*100,2)),str(round(Ngen/Nmap*100,2)),str(Norph),str(round(Norph/Ntot*100,2)),str(round(Norph/Nmap*100,2)),str(Nigen),str(round(Nigen/Ntot*100,2)),str(round(Nigen/Nmap*100,2)))
    return(o)

def run_estimate(tempdir = None,bamfile = None,genefile = None,outputfile = None,infmt = None,threads = 1, verbose=False,orphanbed = None,onlyestimate = True):
    # this function deterines mapping stats and writes them to the tempdir + "/mapping_stats.txt" file.
    statsfile = tempdir + "/mapping_stats.txt"
    if onlyestimate:
        Ntot,Nmap,Ngen,Nigen,Norph = estimate_mapping(tempdir = tempdir,bamfile = bamfile,genefile = genefile,infmt = infmt,threads = threads, verbose =verbose,orphanbed = None)
        old_rep = report_stats(genefile,Ntot,Nmap,Ngen,Nigen,Norph)
        print(old_rep)
    else:
        Ntot,Nmap,Ngen,Nigen,Norph = estimate_mapping(tempdir = tempdir,bamfile = bamfile,genefile = genefile,infmt = infmt,threads = threads, verbose = verbose,orphanbed = None)
        old_rep = report_stats(genefile,Ntot,Nmap,Ngen,Nigen,Norph)
        Ntot,Nmap,Ngen,Nigen,Norph = estimate_mapping(tempdir = tempdir,bamfile = bamfile,genefile = outputfile,infmt = infmt,threads = threads, verbose = verbose,orphanbed = orphanbed)
        new_rep = report_stats(outputfile,Ntot,Nmap,Ngen,Nigen,Norph)
    with open(statsfile,'w') as ofile:
        ofile.write(old_rep + '\n')
        if not onlyestimate:
            ofile.write(new_rep)
    ofile.close()

def summarize_intergenic_diff(file_path):
    """
    Summarize the difference in intergenic read proportions between two sets of results,
    including the file names.

    :param file_path: Path to the file containing the results.
    :return: A string with the difference in intergenic read proportions and file names.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    intergenic_reads = []
    file_names = []
    for i, line in enumerate(lines):
        if "/" in line:
            # Capture the file name
            file_name = line.strip().split(':')[0]
            file_names.append(file_name)
        if 'Intergenic reads' in line:
            # Find the start of the percentage value, assuming the format 'Intergenic reads: NUMBER (total: PERCENTAGE %'
            start = line.find('(total: ') + len('(total: ')
            end = line.find('%', start)
            # Extract the percentage value and convert it to a float
            percentage = float(line[start:end].strip())
            intergenic_reads.append(percentage)

    # Calculate the difference in intergenic read proportions
    if len(intergenic_reads) == 2 and len(file_names) == 2:
        difference = abs(intergenic_reads[0] - intergenic_reads[1])
        return (f"Difference in intergenic read proportions: {difference:.2f}%")
    else:
        return "Could not find two sets of intergenic read proportions or file names."

def report_estimate():
        if do_orphan:
            orphanbed = tempdir + '/orphan_merged.bed'
        else:
            orphanbed = None
        run_estimate(tempdir = tempdir,bamfile = bamfile,genefile = genefile,outputfile = outputfile,infmt = infmt,threads = threads, verbose=verbose,orphanbed = orphanbed,onlyestimate = False)
        # TODO: add parsing of the  tempdir + "/mapping_stats.txt" file. Report nicely. 
        console.print(summarize_intergenic_diff(tempdir + "/mapping_stats.txt"),style = 'bold green')




    # depends on whether it was called only multple files or ont 

def PIPELINE_run_genefile_fix(genefile,infmt,tempdir,verbose,console):
    # Returns the name of the fixed genome annotation file 

    # Given an input file, check if it's missing "gene" and "transcript" features, if so, fix it and output an updated file name 
    features = helper.get_featuretypes(genefile)
    helper.check_gene_exons(genefile,infmt = infmt,verbose = verbose)
    if not 'transcript' in features:
        console.print('Genome annotation warning: Could not find "transcript" features in %s! Trying to fix ...' % genefile,style = 'white')
        if 'mRNA' in features:
            console.print('Found "mRNA" features in %s - renaming as transcripts ... ' % genefile,style = 'white')
            #new_genefile = tempdir + '/' + helper.append_before_ext(os.path.basename(genefile),'fixed')
            new_genefile = tempdir + '/' + 'genome.fixed.' + infmt
            helper.mRNA2transcript(infile = genefile,outfile = new_genefile, verbose = verbose)
            genefile = new_genefile
            if verbose > 0:
                print("Added transcript features. New file name: %s" % genefile)
        else:
            pipeline_error_print("Missing 'transcript' or 'mRNA' features in the genome annotation file %s! Fix genome annotation file (e.g. with gffread)" % genefile)

    if not 'gene' in features:
        console.print('Genome annotation warning: Could not find "gene" features in %s! Trying to fix ...' % genefile,style = 'white')

        #new_genefile = tempdir + '/' + helper.append_before_ext(os.path.basename(genefile),'fixed')
        new_genefile = tempdir + '/' + 'genome.fixed.' + infmt

        #genefilewgenes = tempdir + '/' + genefile.split('/')[-1].replace('.' + infmt,'_addgenes.' + infmt)
        helper.add_gene_features(infile = genefile,outfile = new_genefile,infmt = infmt,verbose = verbose)
        if verbose > 0:
            print('Fix done, annotation with gene features: %s' % new_genefile )
        genefile = new_genefile
        if verbose > 0:
            print("Added gene feature names. New file name: %s" % genefile)
    return(genefile)

########### Main #####################
######################################

if __name__ == "__main__":

    ########### Parse Arguments ################
    ############################################


    # Read the configuration file:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = read_yaml_config('%s/geneext/config.yml' % script_dir)
    helper.print_logo()
    console = Console()
    args = parser.parse_args()
    
    outputfile = args.output
    if not outputfile:
        pipeline_error_print('Please, provide the output file name (-o)!')

    
    bamfile = args.bam
    tempdir = args.t
    
    verbose = int(args.verbose)

    # for console printing 
    if not verbose:
        end = " "
    else:
        end = "\n"
    peaksfile = args.peaks
    genefile = args.genome 
    
    
    maxdist = args.maxdist

    threads = int(args.jobs)
    tag = args.tag
    clip_mode = args.clip_strand

    # peak coverage percentile:
    coverage_percentile = args.peak_perc

    # pipeline execution:--subsamplebam 1000000
    do_macs2 = args.bam is not None  
    do_subsample = args.subsamplebam is not None
    #do_estimate = args.estimate # the option has been removed for the sake of clarity 

    do_clean = not args.keep_intermediate_files
    
    do_orphan = args.orphan
    do_orphan_merge =  do_orphan and not args.nocluster

    ##### Orphan merging defaults:
    # if not set explicitly, will use default values 
    orphan_maximum_distance= int(args.orphan_maxdist) if args.orphan_maxdist else None # DEV
    
    orphan_maximum_size = int(args.orphan_maxsize) if args.orphan_maxsize else None
    
    
    # Clipping 5'-overlaps 
    do_5clip = args.clip_5prime

    # Force 
    do_force = args.force
    # Rerun 
    do_rerun = args.rerun

    # onlyfix
    do_onlyfix = args.onlyfix


    class Logger(object):
        def __init__(self,logfilename):
            self.terminal = sys.stdout
            self.log = open(logfilename, "a")
    
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            # this handles the flush command by doing nothing.
            # you might want to specify some extra behavior here.
            pass 
    # Logging 
    if outputfile:
        logfilename = outputfile + ".GeneExt.log"
    else:
        logfilename = "GeneExt.log"
    sys.stdout = Logger(logfilename)

#################################################################

    do_longest = args.longest # whether to select the longest transcript per gene 
    
#################################################################

    scriptloc = os.path.dirname(os.path.realpath(__file__))
    callcmd = 'python ' + os.path.basename(__file__) + ' '+ " ".join(["-"+str(k)+' '+str(v) for k,v in zip([arg for arg in vars(args)],[getattr(args,arg) for arg in vars(args)]) if v ])
    if verbose:
            print(callcmd)
    # print the values of the pipeine settings


    # If fix_only set - report only doing genome annotation fixes:
    #if config.do_fix_only:
    #    print("CAVE: --onlyfix is set. Only fixing the genome annotation file %s, no extension will be performed!" % (args.genome))

    if do_onlyfix:
        console.print("WARNING: --onlyfix is set. Only fixing the genome annotation file %s, no extension will be performed!" % (args.genome),style = 'bold red')

    console.print(Panel.fit("[bold blue]Preflight checks[/bold blue]", border_style="bold blue"), end = end)

#################################################################
    # check if temporary directory exits;
    if tempdir is None:
        #tempdir = helper.get_prefixed_path(outputfile,prefix = 'tmp_')
        tempdir = config.default_tmp # use the common temp directory name 
        if verbose > 0:
            print("Temporary directory isn't set, setting to %s/" % tempdir)
    else:
        if verbose > 0:
            print("Temporary directory is set to %s/" % tempdir)

    # check if this temporary directory exists:

    if path.isdir(tempdir):
        if verbose > 0:
            print("Temporary directory %s/ found!" % tempdir)
            found_tempdir = True
        if not do_force and not do_rerun:
            pipeline_error_print('Found existing temporary directory %s/.\nDelete the directory or, use --force or --rerun options.' % tempdir)
    else:
        os.mkdir(tempdir)
        if verbose > 0:
            print("Temporary directory created: %s/" % tempdir)
#################################################################
    # if found tempdir, check which files are there 
    
    # subsampling:
    subsampled_bam  = tempdir + '/subsampled.bam' 
    # peak calling 
    macs2_peaks_file = tempdir + '/' + 'allpeaks.bed'
    # peak filtering:
    covfile = tempdir + '/' + 'allpeaks_coverage.bed'
    peaksfilt = tempdir + '/' + 'allpeaks_noov.bed'
    peaksfiltcov = peaksfilt.replace('.bed','_fcov.bed')


    found_subsampled = path.isfile(subsampled_bam)
    found_macs2 = path.isfile(macs2_peaks_file)
    found_covfile = path.isfile(covfile) and path.isfile(peaksfilt) and path.isfile(peaksfilt)

##################################################################
    if not config.do_fix_only and not do_onlyfix:
        # parse input and output formats - run the pipeline accordingly
        if peaksfile is None and bamfile is None:
            pipeline_error_print("Please, specify either alignment [-b] or peaks file [-p]!")

        if bamfile is not None:
            if os.path.isfile(bamfile):
                if verbose > 0:
                    print('Alignment file ... OK')
            else:
                pipeline_error_print('Specified alignment file does not exist!')

    if not config.do_estimate:
        # check output file:
        if outputfile is None and not do_estimate:
            pipeline_error_print('Please, specify the output file [-o]!')

        elif peaksfile is not None:
            if os.path.isfile(peaksfile):
                if verbose > 0:
                    print('Found a peaks file, skipping peak calling ...')
            else:
                pipeline_error_print('Specified peaks file does not exist!\nPlease, specify either a valid .bam file for macs2 or a peaks file.')

        if peaksfile is not None and bamfile is not None:
            pipeline_error_print('Please, specify either a .bam file with reads [-b] or a peaks file [-p] but not both at the same time!')

        if genefile is None:
            pipeline_error_print('Missing genome annotation file [-g]!')

        elif not os.path.isfile(genefile):
            pipeline_error_print('Genome annotation file .... DOES NOT EXIST!')
        else:
            if verbose:
                print('Genome annotation file .... OK')


    else:
        # just check the bam file:
        if bamfile is None:
            pipeline_error_print('Can not estimate mapping without an alignment file!')
    # set temporary directory:
    #if verbose > 0:
    #    print("Temporary directory: %s" % tempdir)
    #if not os.path.exists(tempdir):
    #    os.makedirs(tempdir)
    #    if verbose > 0:
    #            print('Directory created: %s' % tempdir)
    #else:
    #    print('Temporary directory exists. Overwriting!')

    ####################################################
    # Parse input / output formats
    # The output format should also be parsed separately 
    infmt = parse_input_format()
    outfmt = parse_output_format()
    if(outfmt == 'bed'):
        pipeline_error_print("I can't output .bed yet!")
    #compare_infmt_outfmt(infmt,outfmt)
    
    ####################################################
    # Fix the input annotation 
    ####################################################

    if infmt in ['gff','gtf']:
        # check if the fixed version already exists:
        fixed_file_name = tempdir + '/' + genefile.replace('.','.fixed.')
        if do_rerun and os.path.exists(fixed_file_name) and not do_force:
            print('Found fixed genome file: %s' % fixed_file_name)
            genefile = fixed_file_name
        else:
            genefile = PIPELINE_run_genefile_fix(genefile,infmt,tempdir = tempdir,verbose = verbose,console = console)

        if do_longest:
            
            new_genefile = tempdir + '/' + 'genome.fixed.' + infmt
            if verbose:
                print('Selecting the longest transcript per gene ...',end = " ")

            removed_genes_log = tempdir + '/' + 'removed_genes.txt' # file storing the names of the genes removed
            helper.select_longest_transcript(infile = genefile,outfile = new_genefile,infmt = infmt,outfmt = infmt,verbose = verbose,removed_log = removed_genes_log)
            if verbose:
                print('Done selecting the longest transcript per gene.\nRemoved genes written to: %s' % removed_genes_log)
            genefile = new_genefile
            helper.check_file_size(new_genefile,verbose = verbose)


        # Fix 5'overlaps 
        if do_5clip:
            console.print(Panel.fit("[bold blue]5' clipping[/bold blue]", border_style="bold blue"))
            console.print("Clipping 5' overlaps in genes using %s cores..." % str(threads), style = 'white')
            new_genefile = tempdir + '/' + 'genome.fixed.' + infmt
            clip_logfile = new_genefile + '.5clip.log'
            clip_out = helper.clip_5_overlaps(infile = genefile,outfile = new_genefile,logfile = clip_logfile,threads = threads,verbose = verbose)
            console.print("Fixed %s gene five-prime overlaps: %s -> %s\nLog: %s" % (len(clip_out),genefile,new_genefile,clip_logfile))
            genefile = new_genefile
            #helper.check_file_size(genefile,verbose = verbose )

        # Re-order genefile by the order of chromosomes - ISSUE: updates the time stamps 
        if bamfile:
            do_reorder = False
            if do_reorder:
                helper.reorder_by_bam(genefile = genefile,bamfile = bamfile,tempdir = tempdir,verbose = verbose,console = console)
                helper.check_file_size(genefile,verbose = verbose)

    # SJ DEV: what would be an appropriate place to put this function 
    ##################################################
    # Input:
    # splice junction file 
    # Output:
    # a table of gene-to-peak mappings 
    # (this should then become an optional argument for gene extension part of the tool) 

    ##################################################
    # if the SJ file is provided:
    # 1. filter the splice junction file 
    # 2. link genes to DOWNSTREAM peaks using splice junctions   
    # 3. Extend the genes to the most downstream peaks (as a continuous regions?)
    # 4. The rest of the gene extensions should proceed as specified before.   
    ##################################################
        do_sj = False
        if do_sj:
            genic_bed = tempdir + '/' + 'sj.genic.bed'
            helper.get_genic_bed(genefile = genefile,outfile = genic_bed)
            
    #quit() #sj
    if(do_onlyfix):
        console.print(Panel.fit("[bold blue]End of fix[/bold blue]", border_style="bold blue")) 
        quit()
    console.print(Panel.fit("[bold blue]Execution[/bold blue]", border_style="bold blue"))
    ##################################################
    if not config.do_fix_only:
        # parse input file format: 
        if not config.do_estimate_only:    
            #-1. Index input bam 
            if bamfile:
                if not os.path.isfile(bamfile + '.bai'):
                    if verbose > 0:
                        print('Indexing %s' % bamfile)
                    helper.index_bam(bamfile,verbose = verbose,threads=threads)

            # if -m is not set, get a median gene size:
            if not maxdist:
                maxdist = helper.get_quantile_gene_length(inputfile = genefile,fmt = infmt,q = config.gene_maxdist_quant)
                if verbose:
                    print('Maximum allowed extension length is not set, getting median size of the gene - %s bp.' % str(round(maxdist)))
            # if maximum size for orphan peak is not set, set it to the median gene size:
            if do_orphan_merge:
                if not orphan_maximum_size:
                    orphan_maximum_size = round(helper.get_quantile_gene_length(inputfile=genefile,fmt = infmt,q = config.orphan_maxsize_quant))
                    if verbose:
                        print('Maximum size of orphan clusters is set to %s-th quantile of the gene lengths: %s' % (round(config.orphan_maxsize_quant*100),round(orphan_maximum_size)))
                if not orphan_maximum_distance:
                    helper.get_intronic_bed(genefile = genefile,bamfile = bamfile, tempdir = tempdir, verbose = verbose)
                    bedfile = tempdir + '/reg.intronic.bed'
                    orphan_maximum_distance = round(helper.get_bed_length_q(bedfile,config.orphan_maxdist_quant))
                    if verbose:
                        print('Maximum distance between peaks is set to %s-th quantile of intron lengths: %s' % (round(config.orphan_maxdist_quant*100),round(orphan_maximum_distance)))


        # 0. MAPPING - not implemented     
            if config.do_mapping:
                print_task('Running mapping')
                raise(NotImplementedError())
        # 0.1 BAM SUBSAMPLING  
            if do_subsample:
                print_task('Running bam subsampling')
                # check if subsampled
                if do_rerun and found_subsampled and not do_force and verbose > 0:
                    print('Found %s! Skipping subsampling' % subsampled_bam)
                else:
                    nsubs = int(args.subsamplebam)
                    if bamfile:
                        if not os.path.isfile(bamfile + '.bai'):
                            if verbose > 0:
                                print('Indexing %s' % bamfile)
                            helper.index_bam(bamfile,verbose = verbose,threads=threads)
                    # check here if it's an integer
                    helper.subsample_bam(inputbam = bamfile,outputbam = subsampled_bam,nreads = nsubs,verbose = verbose,threads=threads)
                    # now, replace for downstream:
                    if verbose > 0:
                        print('Indexing %s' % bamfile)
                    helper.index_bam(subsampled_bam,verbose = verbose,threads=threads)
                    if verbose > 0:
                        print('Subsampling done.')
                    bamfile = subsampled_bam
                    console.print('done',style = 'bold green')

        # 1. MACS2
            if do_macs2:
                print_task('Running macs2')
                peaksfile = tempdir + '/' + 'allpeaks.bed'
                if do_rerun and found_macs2 and not do_force:
                    if verbose > 0:
                        print('Found %s! Skipping peak calling with MACS2.' % peaksfile)
                    console.print('done',style = 'bold green')
                else:
                    helper.split_strands(bamfile,tempdir,verbose = verbose,threads = threads)
                    helper.run_macs2(tempdir+'/' + 'plus.bam','plus',tempdir,verbose = verbose)
                    helper.run_macs2(tempdir+'/' + 'minus.bam','minus',tempdir,verbose = verbose)
                    helper.collect_macs_beds(outdir = tempdir,outfile = peaksfile,verbose = verbose)
                    # reorder by the genome file 
                    chrnamesfile = tempdir + '/chr_names.txt'
                    #peaskfile_ordered = peaksfile.replace('.bed','.ordered.bed')
                    helper.get_chr_names(bamfile, chrnamesfile,verbose = verbose)
                    helper.order_bed(peaksfile,peaksfile,chrnamesfile,verbose = verbose)
                    # fix peaks with 0-start coordinate 
                    helper.fix_bed_start(peaksfile,peaksfile,verbose = verbose)
                    console.print('done',style = 'bold green')
            else:
                print('Skipping macs2. Running gene extension with %s and %s.' % (peaksfile,genefile))

        # 3. If used macs to call the peaks, filter the peaks by the coverage:
        # REPLACE WITH A PIPELINE FUNCTION 
  
            if do_macs2:    
                #console.print('======== Filtering macs2 peaks =================',style = 'bold yellow',end = end)
                print_task('Filtering macs2 peaks')

                if do_rerun and found_covfile and not do_force:
                    if verbose > 0:
                        print('Found %s! Skipping peak filtering.' % covfile)
                    console.print('done',style = 'bold green')
                else:
                    # compute coverage for all the peaks:
                    # check if bam file is indexed:
                    if not os.path.isfile(bamfile + '.bai') and 'subsample' in bamfile:
                        if verbose > 0:
                            print('Indexing %s' % bamfile)
                        helper.index_bam(bamfile,verbose = verbose,threads=threads)
                    if verbose > 0:
                        print('Computing coverage ...')
                    helper.get_coverage(inputbed_a=peaksfile,input_bam = bamfile,outputfile = covfile,verbose = verbose,mean = config.mean_coverage,threads = threads)
                    # get the peaks overlapping genes:
                    if verbose > 0:
                        print('Getting genic peaks ...')
                    genicpeaksfile = tempdir + '/genic_peaks.bed'
                    helper.intersect(inputbed_a = covfile, inputbed_b = genefile,outputbed=genicpeaksfile,by_strand = True,verbose = verbose)
                    nrow = helper.get_tsv_nrow(genicpeaksfile)
                    if nrow == 0:
                        pipeline_error_print("No lines in %s" % genicpeaksfile)
                    if coverage_percentile == 0:
                        if verbose > 0:
                            print('Coverage percentile is not set - retaining all the peaks...')
                        count_threshold = 0
                    else:
                        # get coverage percentile from genic peaks:
                        count_threshold = helper.get_coverage_percentile(inputfile = genicpeaksfile,percentile = coverage_percentile,verbose = verbose)
                        # hell, you also have to replace the column 
                        if verbose > 0:
                            print('%s-th %s coverage percentile for %s is %s %s. Filtering out the peaks below this value...' % (coverage_percentile,'mean' if config.mean_coverage else '' ,genicpeaksfile,str(count_threshold),'reads/base' if config.mean_coverage else 'reads'))
                    
                    # get peaks not overlapping the genes and filter them by coverage
                    peaksfilt = tempdir + '/' + 'allpeaks_noov.bed'
                    peaksfiltcov = peaksfilt.replace('.bed','_fcov.bed')
                    if verbose > 0:
                        print('Removing peaks overlapping genes ... => %s' % peaksfilt)
                    # f=1 will filter out only the peaks fully contained within genes!!!
                    # This means, that non-fully contained peaks will be used for gene extension. 
                    helper.outersect(inputbed_a = covfile,inputbed_b = genefile,outputbed=peaksfilt,by_strand = True, verbose = verbose,f = config.peak_overlap_fmin)
                    peakfilt_message = helper.filter_by_coverage(inputfile = peaksfilt,outputfile = peaksfiltcov,threshold = count_threshold,verbose = verbose, do_message = True)
                    console.print(peakfilt_message,style = 'bold yellow')
                    peaksfilt = peaksfiltcov
                    console.print('done',style = 'bold green')
            else:
                print('No macs2 run specified, simple filtering the peaks overlapping the genes.')
                # Simply remove peaks overlapping genes:
                peaksfilt = tempdir + '/' + 'allpeaks_noov.bed'
                if verbose > 0:
                    print('Removing peaks overlapping genes ... => %s' % peaksfilt)
                #outersect_peaks(genefile = genefile, peaksfile = peaksfile, outputbed = peaksfilt, verbose = verbose)
                helper.outersect(inputbed_a = peaksfile,inputbed_b = genefile,outputbed=peaksfilt,by_strand = True, verbose = verbose)
        # 3. Extend genes 
            #console.print('======== Extending genes =======================',style = 'bold yellow',end = end)
            print_task('Extending genes')
            helper.extend_genes(genefile = genefile,peaksfile = peaksfilt,outfile = outputfile,maxdist = int(maxdist),temp_dir = tempdir,verbose = verbose,extension_mode = config.extension_mode,infmt = infmt,outfmt = outfmt,tag = tag,clip_mode = clip_mode,write_original_transcript = config.write_original_transcript)
            console.print('done',style = 'bold green')

        # 4. Add orphan peaks
            if do_orphan:
                #console.print('======== Adding orphan peaks ===================',style = 'bold yellow', end = end)run_
                #run_orphan(infmt = infmt,outfmt = outfmt,verbose = verbose,merge = do_orphan_merge)
                run_orphan()
                
            if config.do_report:
                #print('======== Creating PDF report =======================')
                print_task('Creating PDF report')
                raise(NotImplementedError())
                generate_report()

        # 5. Estimate intergenic mapping
            if config.do_estimate:
                console.print('======== Estimating intergenic mapping =========',style = 'bold yellow')
                if bamfile:
                    if do_orphan:
                        run_estimate(tempdir = tempdir,bamfile = bamfile,genefile = genefile,outputfile = outputfile,infmt = infmt,threads = threads, verbose=verbose,orphanbed = tempdir + '/orphan_merged.bed',onlyestimate = False)
                    else:
                        run_estimate(tempdir = tempdir,bamfile = bamfile,genefile = genefile,outputfile = outputfile,infmt = infmt,threads = threads, verbose=verbose,orphanbed = None,onlyestimate = False)
                else:
                    print("No bamfile specified - omitting mapping estimation")
        elif bamfile:
            print('--estimate is set. Estimating mapping rates for %s with %s' % (genefile,bamfile))
            run_estimate(tempdir = tempdir,bamfile = bamfile,genefile = genefile,outputfile = outputfile,infmt = infmt,threads = threads, verbose=verbose,orphanbed = None,onlyestimate = True)
        console.print(Panel.fit(Text("All done!", style="bold blue"), border_style="bold blue"))

        # Report extensions
        if do_orphan:
            if do_orphan_merge:
                orphan_bed = tempdir + '/orphan_merged.bed'
            else:
                orphan_bed = tempdir + '/orphan.bed'
        else:
            orphan_bed = None
        report_extensions(file_path = tempdir+'/extensions.tsv',orphan_bed = orphan_bed,n_genes=helper.get_number_of_genes(genefile,fmt = infmt))
        
        if config.do_plots:
            helper.plot_extensions(infile = tempdir+'/extensions.tsv',outfile = outputfile + '.extension_length.pdf',verbose = verbose)
            helper.plot_peaks(genic = tempdir + '/genic_peaks.bed',noov = tempdir + '/allpeaks_noov.bed',outfile = outputfile + '.peak_coverage.pdf',peak_perc = coverage_percentile,verbose=verbose)

        if config.do_estimate:
            report_estimate()
        if do_clean:
            # clean big files
            clean_tmp(tempdir = tempdir)
