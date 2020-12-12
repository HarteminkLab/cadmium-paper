
import pandas as pd
import numpy as np
from datasets import read_orfs_data
from src.reference_data import read_park_TSS_PAS, read_sgd_orfs
from src.utils import print_fl


def determine_paper_set(orfs, TSSs, half_lives, mnase_coverage):

    # ORFs list
    orfs = orfs.copy()

    def _filter_orfs(old_orfs, orfs):
        return orfs, "%d (%d)" % (len(orfs), (len(orfs)-len(old_orfs)))

    print_fl("SacCer3:                    %d" % len(orfs), log=True)

    orfs, summary = _filter_orfs(orfs, 
                        orfs[(orfs['orf_class'] == 'Uncharacterized') | 
                        (orfs['orf_class'] == 'Verified')])
    print_fl("Verified/Uncharacterized:   %s" % summary, log=True)


    orfs, summary = _filter_orfs(orfs, 
                                 orfs[(orfs['length'] >= 500)])
    print_fl("ORFs with length >500 nt:   %s" % summary, log=True)

    orfs, summary = _filter_orfs(orfs, orfs[~np.isnan(orfs['TSS'])])
    print_fl("Has Park TSS:               %s" % summary, log=True)

    orfs, summary = _filter_orfs(orfs, orfs.join(half_lives[['half_life']], how='inner'))
    print_fl("Has half life               %s" % summary, log=True)
    
    cutoff = 0.85
    mnase_coverage = mnase_coverage[mnase_coverage['coverage'] > cutoff]
    orfs, summary = _filter_orfs(orfs, orfs.join(mnase_coverage, how='inner'))
    print_fl(">%d%% MNase-seq coverage:    %s" % (int(cutoff*100), summary), log=True)

    print_fl("", log=True)
    print_fl("Using %d ORFs" % len(orfs), log=True)
    return orfs

    

def filter_orfs(orfs, start_key, stop_key, chrom, start, stop, 
    strands=['+', '-']):
    """
    Filter orfs based on if start and end keys are within requested
    start and end span
    """

    same_chrom = orfs.chr == chrom
    within_start = (orfs[start_key] >= start) & (orfs[start_key] <= stop)
    within_stop = (orfs[stop_key] >= start) & (orfs[stop_key] <= stop)
    cor_strand = orfs.strand.isin(strands)

    return orfs[same_chrom & within_start & within_stop & cor_strand]


def find_adjacent_orfs(query_orfs):

    query_orfs = query_orfs.copy()

    query_orfs['upstream_tandem'] = None
    query_orfs['upstream_divergent'] = None
    query_orfs['downstream_tandem'] = None
    query_orfs['downstream_convergent'] = None

    search_length = 500

    for chrom in range(1, 17):
        print_fl(chrom)
        chrom_orfs = query_orfs[query_orfs.chr == chrom]
        for orf_name, orf in chrom_orfs.iterrows():
            if orf.strand == '+':
                same_strand = '+'
                antisense_strand = '-'
                upstream_key = 'transcript_stop'
                downstream_key = 'transcript_start'

                upstream_span = orf.transcript_start-search_length, orf.transcript_start
                downstream_span = orf.transcript_stop, orf.transcript_stop+search_length
            else:
                same_strand = '-'
                antisense_strand = '+'
                upstream_key = 'transcript_start'
                downstream_key = 'transcript_stop'

                upstream_span = orf.transcript_stop, orf.transcript_stop+search_length
                downstream_span = orf.transcript_start-search_length, orf.transcript_start

            # find upstream ORFs
            found_tandem_upstream = filter_orfs(chrom_orfs, upstream_key, 
                upstream_key, orf.chr, upstream_span[0], upstream_span[1], 
                strands=[same_strand])
            found_divergent_upstream = filter_orfs(chrom_orfs, upstream_key, 
                upstream_key, orf.chr, upstream_span[0], upstream_span[1], 
                strands=[antisense_strand])

            # find downstream ORFs
            found_convergent_downstream = filter_orfs(chrom_orfs, 
                downstream_key, downstream_key, orf.chr, 
                downstream_span[0], downstream_span[1], 
                strands=[antisense_strand])
            found_tandem_downstream = filter_orfs(chrom_orfs, downstream_key, 
                downstream_key, orf.chr, downstream_span[0], 
                downstream_span[1], strands=[same_strand])

            tot = len(found_tandem_upstream) + len(found_divergent_upstream) + \
                  len(found_convergent_downstream) + len(found_tandem_downstream)
                
            query_orfs.loc[orf_name, 'upstream_tandem'] = ','.join(found_tandem_upstream.index.values)
            query_orfs.loc[orf_name, 'upstream_divergent'] = ','.join(found_divergent_upstream.index.values)
            query_orfs.loc[orf_name, 'downstream_tandem'] = ','.join(found_tandem_downstream.index.values)
            query_orfs.loc[orf_name, 'downstream_convergent'] = ','.join(found_convergent_downstream.index.values)

    for key in ['upstream_tandem', 'upstream_divergent', 
            'downstream_tandem', 'downstream_convergent']:
        query_orfs.loc[query_orfs[key] == '', key] = np.nan

    return query_orfs
