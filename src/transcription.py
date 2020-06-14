
import numpy as np
import pandas


def filter_rna_seq(rna_seq, start=None, end=None, chrom=None,
    time=None, strand=None):
    """
    Filter RNA-seq data given the argument parameters, do not filter if
    not specified
    """

    if strand is not None:
        select = rna_seq.strand == strand
    else:
        # dummy select
        select = ~(rna_seq.strand == None)

    if time is not None:
        select = select & (rna_seq.time == time)

    if chrom is not None:
        select = (rna_seq.chr == chrom) & select
 
    if start is not None and end is not None:
        select = ((rna_seq.stop <= end) & 
                  (rna_seq.start >= start)) & select

    return rna_seq[select].copy()


def filter_rna_seq_pileup(pileup, start=None, end=None, chrom=None,
    time=None, strand=None):
    """
    Filter RNA-seq data given the argument parameters, do not filter if
    not specified
    """

    if strand is not None:
        select = pileup.strand == strand
    else:
        # dummy select
        select = ~(pileup.strand == None)

    if time is not None:
        select = select & (pileup.time == time)

    if chrom is not None:
        select = (pileup.chr == chrom) & select
 
    if start is not None and end is not None:
        select = ((pileup.position < end) & 
                  (pileup.position >= start)) & select

    return pileup[select].copy()


def calculate_reads_TPM(orfs, rna_seq, 
    times=[0.0, 7.5, 15.0, 30.0, 60.0, 120.0], antisense=False):
    """Get RNA-seq read counts and TPM"""

    read_counts = orfs[[]].copy()

    for time in times:
        read_counts[time] = 0

    for chrom in range(1, 17):

        chrom_orfs = orfs[orfs.chr == chrom]
        chrom_rna_seq = filter_rna_seq(rna_seq, chrom=chrom)
        
        for idx, orf in chrom_orfs.iterrows():
            orf_rna_seq = filter_rna_seq(chrom_rna_seq, start=orf.start, end=orf.stop)

            for time in times:
                time_rna_seq = filter_rna_seq(orf_rna_seq, time=time)

                if antisense:
                    strand_select = (time_rna_seq.strand != orf.strand)
                else:
                    strand_select = (time_rna_seq.strand == orf.strand)

                cur_orf_read_counts = time_rna_seq[strand_select]

                read_counts.loc[idx, time] = len(cur_orf_read_counts)

    orf_TPMs = convert_to_TPM(read_counts, orfs)

    return read_counts, orf_TPMs


def convert_to_TPM(sense_reads, orfs, times=[0.0, 7.5, 15.0, 30.0, 60.0, 120.0]):
    """Convert to TPM, normalize by length of ORF, then divide 
    by sum of each time, multiply by 1 mil so each time point 
    sums to 1e6"""
    data = sense_reads.join(orfs[['length']])

    # normalize by ORF length
    for time in times:
        data.loc[:, time] = data[time] / data['length']
    data = data[times]

    scaling = data.sum()
    
    data = (data / scaling * 1e6) # scale to 1 million
    return data


def calculate_xrates(tpm, half_lives, times=[0.0, 7.5, 15.0, 30.0, 60.0, 120.0]):
    # calculate the transcription rates for each ORF
    orf_xrates = tpm.join(half_lives[['half_life']], how='inner')
    for orf, row in orf_xrates.iterrows():
        xrates = calculate_xrate(row[times], row.half_life)
        orf_xrates.loc[orf, times] = xrates
    return orf_xrates

def calculate_xrate(data, t_half=None, times=[0.0, 7.5, 15.0, 30.0, 60.0, 120.0]):
    """
    Calculate transcription rates given a gene with known half life
    and mRNA transcripts.

    Calculate steady state transcription using differential equation:
        dC/dt = 0 = R - kC

    Calculate consecutive time points using solved differential equation:
        C = R/k + G*exp[-kt]

    Then solve difference equation for R and G to connect consecutive concentrations:

        C2 = R/k + G * exp[-k(t2-t1)]
        C1 = R/k + G * exp[-k(t1-t1)]
    """

    data = data.copy()
    def conc_eq(t, R, k, G):
        return R/k + G * np.exp(-k*t)

    if t_half is None:
        t_half = data.half_life

    k = np.log(2)/t_half

    C = data[times].values
    t = times

    rate_parameters = {'C1': [], 'C2':[], 'k': [],'t1':[], 't2':[], 'R':[], 'G': []}

    for i in range(0, len(C)):

        t2 = t[i]
        C2 = C[i]

        # steady state equation
        if i == 0:
            t1 = t2
            C1 = C2
            R1 = C2*k
            G1 = C2-R1/k

        # difference equation
        else:
            t1 = t[i-1]
            C1 = C[i-1]

            e_cur = np.exp(-k*(t2-t1))
            R1 = (C2 - e_cur*C1)*k / (1 - e_cur)
            G1 = C1-R1/k

        rate_parameters['C1'].append(C1)
        rate_parameters['C2'].append(C2)
        rate_parameters['R'].append(R1)
        rate_parameters['G'].append(G1)
        rate_parameters['t1'].append(t1)
        rate_parameters['t2'].append(t2)
        rate_parameters['k'].append(k)
        data[times[i]] = R1

    return data


def _conc_eq(t, R, k, G, t1):
    """Concentration equation given time, decay rate, growth and starting time"""
    return R/k + G * np.exp(-k*(t-t1))