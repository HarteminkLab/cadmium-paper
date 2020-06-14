
from src.utils import mkdirs_safe
from src.utils import print_fl
from src.timer import Timer
from config import *

# global timer
timer = Timer()

rna_bam_dir = '%s/rna_seq' % BAM_DIR
mnase_bam_vt = '%s/mnase_seq/vt' % BAM_DIR
mnase_bam_hm = '%s/mnase_seq/hm' % BAM_DIR

def download_bam():
    pass

def init_mnase_seq():

    from src.read_bam import read_mnase_set

    vt_filenames = ['DM498_sacCer3_m1_2020-05-20-17-18.bam',
                    'DM499_sacCer3_m1_2020-05-20-17-32.bam',
                    'DM500_sacCer3_m1_2020-05-20-17-51.bam',
                    'DM501_sacCer3_m1_2020-05-20-18-06.bam',
                    'DM502_sacCer3_m1_2020-05-20-18-20.bam',
                    'DM503_sacCer3_m1_2020-05-20-18-32.bam']

    hm_filenames = ['DM504_sacCer3_m1_2020-05-20-18-48.bam',
                    'DM505_sacCer3_m1_2020-05-20-19-05.bam',
                    'DM506_sacCer3_m1_2020-05-20-19-21.bam',
                    'DM507_sacCer3_m1_2020-05-20-19-38.bam',
                    'DM508_sacCer3_m1_2020-05-20-19-57.bam',
                    'DM509_sacCer3_m1_2020-05-20-20-17.bam']

    # Read VT BAM files
    print_fl("Reading VT MNase-seq BAM...", end='')
    vt_mnase = read_mnase_set(mnase_bam_vt, vt_filenames, 'vt_dm498_503', 
        debug=DEBUG)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    print_fl("Reading HM MNase-seq BAM...", end='')
    hm_mnase = read_mnase_set(mnase_bam_hm, hm_filenames, 'hm_dm504_509', 
        debug=DEBUG)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # depth of each dataset
    vt_depth = vt_mnase[['chrom', 'time']].groupby('time').count()\
        .rename(columns={'chrom':'count'})
    hm_depth = hm_mnase[['chrom', 'time']].groupby('time').count()\
        .rename(columns={'chrom':'count'})

    print_fl("VT read depth:\n" + str(vt_depth), end='\n\n')
    print_fl("HM read depth:\n" + str(hm_depth), end='\n\n')

    from src.chromatin import sample_mnase
    sample_vt_depth = vt_depth['count'].min()
    sample_hm_depth = hm_depth['count'].min()

    print_fl("Sampling VT to %d for each time point..." % 
        sample_vt_depth, end='')
    vt_mnase_sampled = sample_mnase(vt_mnase, sample_vt_depth)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    print_fl("Sampling HM to %d for each time point..." % 
        sample_hm_depth, end='')
    hm_mnase_sampled = sample_mnase(hm_mnase, sample_hm_depth)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Merge VT and HM datasets
    print_fl("Merging VT and HM...", end='')
    merged_mnase_sampled = vt_mnase_sampled.append(hm_mnase_sampled)
    merged_mnase_sampled = merged_mnase_sampled[['chrom', 'start', 
    'stop', 'length', 'mid', 'time', 'source']].sort_values(
        ['source', 'time', 'chrom', 'start'])
    merged_mnase_sampled = merged_mnase_sampled.rename(columns={'chrom':'chr'})
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Save to disk
    save_path = '%s/mnase_seq_merged_dm498_509_sampled.h5.z' % mnase_dir
    print_fl("Saving merged MNase-seq to %s..." % save_path, end='')
    merged_mnase_sampled.to_hdf(save_path, 'mnase_data', mode='w', complevel=9,
        complib='zlib')
    print_fl("Done.")
    timer.print_time()
    print_fl()

def init_rna_seq():

    rna_seq_filenames = ['DM538_sacCer3.bam',
                         'DM539_sacCer3.bam',
                         'DM540_sacCer3.bam',
                         'DM541_sacCer3.bam',
                         'DM542_sacCer3.bam',
                         'DM543_sacCer3.bam']

    # Read RNA-seq BAM
    from src.read_bam import read_rna_seq_set
    print_fl("Reading RNA-seq BAM...", end='')
    rna_seq = read_rna_seq_set(rna_bam_dir, rna_seq_filenames, debug=DEBUG)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Save RNA-seq data frame
    save_path = rna_seq_path
    print_fl("Saving RNA-seq to %s..." % save_path, end='')
    rna_seq.to_hdf(save_path,
    'rna_seq_data', mode='w', complevel=9, complib='zlib')
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # convert to pileup dataframe
    mkdirs_safe([pileup_chrom_dir])

    print_fl("Calculating RNA-seq pileup...", end='')
    from src.rna_seq_pileup import calculate_rna_seq_pileup
    pileup = calculate_rna_seq_pileup(rna_seq, timer)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    save_path = '%s/rna_seq_pileup_dm538_543.h5.z' % rna_dir
    print_fl("Saving RNA-seq pileup to %s..." % save_path, end='')
    pileup.to_hdf(save_path,
        'pileup', mode='w', complevel=9, complib='zlib')
    print_fl("Done.")
    timer.print_time()
    print_fl()

def main():
    """
    Initial BAM data download and conversion to dataframes

    1. Download RNA-seq and MNase-seq BAM files to disk
    2. Read RNA-seq BAM convert to dataframe and save
    3. Compute RNA-seq pileup and save
    4. Read MNase-seq BAM duplicates
    5. Merged, sample, and save

    Inputs:
    - output directory path
    - RNA-seq BAM files path
    - MNase-seq BAM files path

    Output:
    - RNA-seq data frame of reads
    - RNA-seq pileup data frame
    - MNase-seq merged data frame

    """

    print_fl("*******************************")
    print_fl("* 1      Initialization       *")
    print_fl("*******************************")

    # ------ Setup -------------

    print_preamble()


    # Make directories
    mkdirs_safe([
        rna_bam_dir, mnase_bam_hm, mnase_bam_vt,
        rna_dir, mnase_dir
        ])

    # ------- Download BAM files to disk ------------

    # TODO: download files to RNA and MNase-seq dirs
    # when GEO submission is complete
    # Assume for now BAM files have been downloaded
    # and correctly in sub folders
    # download_bam()

    print_fl("\n------- RNA-seq ----------\n")
    init_rna_seq()

    print_fl("\n------- MNase-seq ----------\n")
    init_mnase_seq()

    print_fl("Data initialization done. Time elapsed: %s" % timer.get_time())

if __name__ == '__main__':
    main()
