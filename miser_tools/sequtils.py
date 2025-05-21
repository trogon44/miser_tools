import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Bio import SeqIO, SeqRecord, AlignIO
import pysam
import os
import seaborn as sns
from tqdm import tqdm
import re
from Levenshtein import distance as levenshtein_distance
import subprocess

def run_sys_command(command):
    print(f'Running command: {command}')
    result = subprocess.run(
        command, 
        shell=True, 
        stdout=subprocess.DEVNULL,  # Suppress stdout
        stderr=subprocess.PIPE      # Capture stderr, if needed
    )
    
    # Check the return code for success or failure
    if result.returncode == 0:
        print("Command executed successfully.")
    else:
        print("Command failed. Error:", result.stderr.decode())
    return


class BaseSeqs:
    # initial BaseSeqs class with reference fasta, fastq files of reads, and the start and stop indices of Cas protein
    def __init__(self, ref_fasta, reads_fastq, casStart, casEnd):
        self.ref_fasta = Path(ref_fasta)
        self.reads_fastq = Path(reads_fastq)
        self.casStart = casStart
        self.casEnd = casEnd
        self.fastq_index = SeqIO.index(str(self.reads_fastq), 'fastq')
        self.read_count = len(list(self.fastq_index.keys()))
        self.ref_seqrecord = SeqIO.read(str(self.ref_fasta), 'fasta')

        print(f'Indexing fastq: "{str(self.reads_fastq)}"')
        print(f'Fastq contains {self.read_count} reads.')

    # generate fastq info dataframe (and plot jointplot if desired)
    def fastq_info(self, makeplot=True):
        """"

        Get some info about the fastq dataset.

        Usage:
        fastq_info(makeplot=True)

        Description:
        This gets some information on the fastq read dataset and includes a visualization
        of read length and quality if makeplot=True

        Parameters
        -----------------------------------------------------
        makeplot: boolean, display scatterplot of length vs. mean q-score if True

        Returns
        -----------------------------------------------------
        None
        But it will build a dataframe of read length and average q-scores and a dictionary of
        summary statistics.

        """
        read_info = dict()
        for read in self.fastq_index.keys():
            record = self.fastq_index[read]
            read_info[read] = {'length': len(record.seq), 'mean_q': np.mean(record.letter_annotations['phred_quality'])}
        self.fastqinfo = pd.DataFrame.from_dict(read_info, orient='index')

        if makeplot:
            sns.jointplot(data=self.fastqinfo, x='length', y='mean_q', s=5)
            plt.show()

        self.fastq_summary = {'total reads': len(self.fastqinfo),
                            'avg. length': np.mean(self.fastqinfo['length']),
                            'avg. q-score': np.mean(self.fastqinfo['mean_q'])}
        
    # filter reads according to selection criteria and create pairwise alignment dictionary with minimap2
    def align2ref(self, full_plasmid = True, qscore_cutoff = 20, length_cutoff = 1000, map_qual_cutoff = 21):
        print(f'Filtering fastq for sequences with mean Q-score > {qscore_cutoff} and length > {length_cutoff}.')
        # check if fastqinfo exists. if not, create it.
        if not hasattr(self, 'fastqinfo'):        
            self.fastq_info(makeplot=False)
        # get the read names that pass the filter criteria
        if full_plasmid:
            filter_keys = self.fastqinfo[(self.fastqinfo['length'] > length_cutoff) & (self.fastqinfo['mean_q'] > qscore_cutoff)].index
        else:
            # get the read names that pass the filter criteria
            filter_keys = self.fastqinfo[(self.fastqinfo['mean_q'] > qscore_cutoff)].index

        # generate the sequence records list for the reads passing the filter criteria and write a temporary fastq
        filt_seqs = [self.fastq_index[k] for k in filter_keys]
        tempfastq = 'filt_seqs_temp.fastq'
        SeqIO.write(filt_seqs, tempfastq, 'fastq')
        sam_temp = "temp.sam"

        if full_plasmid:
            # create a doubled vector reference fasta for minimap2 to gracefully handle the circular wraparound problem
            double_vec_name = self.ref_fasta.stem + "_double" + self.ref_fasta.suffix
                        
            # create and write a doubled version of the vector so that all fastq reads will be able to fully align inside
            seq_rec_double = SeqRecord.SeqRecord(
                                seq = self.ref_seqrecord.seq*2,
                                id = self.ref_fasta.stem + "_double",
                                name = self.ref_fasta.stem + "_double",
                                description = f"doubled version of {self.ref_fasta.stem} for alignment")

            SeqIO.write(seq_rec_double, double_vec_name, 'fasta')

            # build call to minimap2 to map the fastq query onto the doubled vector
            minimap2_call = f'minimap2 -ax map-ont {double_vec_name} {str(self.reads_fastq)} -o {sam_temp}'

            print("Running minimap2 on doubled vector.")
        else:
            # build call to minimap2 to map the fastq query onto the doubled vector
            minimap2_call = f'minimap2 -ax map-ont {self.ref_fasta} {str(self.reads_fastq)} -o {sam_temp}'
            print("Running minimap2 on amplicon sequence.")



        print(minimap2_call)
        stderr = os.system(minimap2_call)
        # exit with error if minimap2 returns abnormally
        if stderr != 0:
            raise OSError("Minimap2 command failed with abnormal exit status.")

        print("Finished with alignment.")
        print("Parsing alignments and generating pairwise alignment keys.")

        # use pysam to read in the alignment file generator
        alignment_set = pysam.AlignmentFile(sam_temp)
    
        # initialize the dictionary that will hold the pairwise alignments for each unique fastq
        pairwise_dict = dict()
        
        # iterate over all alignments in the alignment set
        for alignment in alignment_set:
            # skip if the alignment isn't mapped to the vector or the mapping quality falls below map_qual_cutoff
            try:
                unmapped = not alignment.is_mapped
            except:
                unmapped = alignment.is_unmapped

            if (unmapped) or (alignment.mapping_quality < map_qual_cutoff) or (alignment.is_secondary):
                continue
            
            # set name to the query name for the alignment (note that this might not be unique for alignment)
            name = alignment.query_name
            # get the aligned pairs tuple vector as a numpy array
            pairwise = np.array(alignment.get_aligned_pairs(matches_only=True))
            qual_score = [ord(c)-33 for c in alignment.qqual]
            align_info = {'pairwise': pairwise,
                          'mapping_quality': alignment.mapping_quality,
                          'mean_q': np.mean(qual_score),
                          'reverse': alignment.is_reverse}
            # append the pairwise array to the dictionary with the key "name". (The setdefault stuff initializes an empty
            # list in the event that the entry doesn't exist but otherwise appends if it does.)
            pairwise_dict.setdefault(name,[]).append(align_info)
            
        # close the alignment file   
        alignment_set.close()
        # remove any temporary files that were created. This uses "try" since not all these files
        # may be written in certain circumstances.
        try:
            os.remove(sam_temp)
            os.remove(tempfastq)
            os.remove(double_vec_name)
        except:
            pass

        self.pairwise_dict = pairwise_dict

    def find_barcodes(self, nanopore_diagnost = False):
        """"

        Get the barcodes from fastq of reads.

        Usage:
        find_barcodes()

        Description:
        This function finds the barcode sequences from every read. The input reference fasta must
        contain a stretch of N's in place of the expected barcode. The code will grab the read
        sequence corresponding to the stretch of N's.

        Parameters
        -----------------------------------------------------
        None

        Returns
        -----------------------------------------------------
        None
        But it will build a dataframe containing every read ID and its associated barcode

        """
        # check if reference fasta contains a sensible barcode (ie. a contiguous stretch of Ns)
        barcode_start = self.ref_seqrecord.seq.find("N")
        barcode_end = self.ref_seqrecord.seq.rfind("N")
        Ncount = np.sum([pos.upper() == 'N' for pos in self.ref_seqrecord.seq])

        ambiguous_barcode = (barcode_end - barcode_start + 1) != Ncount
        if ambiguous_barcode:
            raise ValueError("Non-contiguous barcode (or unexpected N's in the sequence)")
        
        # check that the pairwise alignment dictionary exists
        if not hasattr(self, 'pairwise_dict'):
            raise ValueError("Must run align2ref method before find_gaps.")


        barcode_dict = dict()
        for read_name in list(self.pairwise_dict.keys()):
        
            read_reference = np.mod(self.pairwise_dict[read_name][0]['pairwise'][:,1], len(self.ref_seqrecord.seq))
            bc_start_ind = np.where(read_reference == barcode_start)[0]
            bc_end_ind = np.where(read_reference == barcode_end)[0]
            query_start = self.pairwise_dict[read_name][0]['pairwise'][bc_start_ind,0]
            query_end = self.pairwise_dict[read_name][0]['pairwise'][bc_end_ind,0]
            fastq_entry = self.fastq_index[read_name]
            if (len(query_start) == 1) & (len(query_end) == 1):
                if np.abs(query_end[0] - query_start[0]) <= 25:
                    label = 'valid'
                    if self.pairwise_dict[read_name][0]['reverse']:
                        query_seq = fastq_entry.seq.reverse_complement()
                    else:
                        query_seq = fastq_entry.seq
                    bc = str(query_seq[query_start[0]: query_end[0] + 1])
                
                else:
                    label = 'too_long'
                    bc = np.nan
            else:
                label = 'missing'
                bc = np.nan

            if nanopore_diagnost:
                read_des = fastq_entry.description
                read_ch_match = re.search(r'read=(\d+).*ch=(\d+)', read_des)
                
                barcode_dict[read_name] = {'barcode':bc, 
                                        'label':label, 
                                        'channel': int(read_ch_match.group(2)),
                                        'read': int(read_ch_match.group(1)),
                                        'reverse': self.pairwise_dict[read_name][0]['reverse']}
            else:
                barcode_dict[read_name] = {'barcode':bc, 
                                        'label':label, 
                                        'reverse': self.pairwise_dict[read_name][0]['reverse']}

        self.barcodes = pd.DataFrame.from_dict(barcode_dict, orient='index')

    def estimate_library_size(self, use_preseq = False):
        """"

        Estimate the total number of unique barcodes in the library.

        Usage:
        estimate_library_size(use_preseq=False)

        Description:
        This function computes estimates for the number of unique barcodes occurring in the library.
        It will always make a simple estimate based on an assumption of a uniform library (i.e. 
        all of the barcodes are present with equal likelihood). Barcodes are assumed to be identical
        if they lie within a Levenshtein distance of 3 of one another.

        If use_preseq==True it will try to use the more sophisticated library size estimate from
        the preseq software. However, for this to work, there must be some barcodes observed six
        or more times which will often not be the case for a large and well balanced library.
        If the library fails to meet these requirements the code will report that but not error.

        Parameters
        -----------------------------------------------------
        use_preseq: boolean, whether to use preseq or not

        Returns
        -----------------------------------------------------
        None
        But it will build the dictionary self.lib_size which will contain the size estimates.

        """
        if not hasattr(self, 'barcodes'):
            raise ValueError("Must run find_barcodes before estimating library size.")

        # find recurring barcodes with options for duplex filtering and Levenshtein distance cutoff threshold for matches
        def find_recurring_barcodes(barcode_df, filter_duplex=True, lev_cutoff = 3):
            # get a dataframe with only those having good barcodes
            valid_barcodes = barcode_df[barcode_df['label'] == 'valid']
            # initialize a matrix to populate with the Levenshtein distances. It is filled with "lev_cutoff" +1
            ld_matrix = np.ones((len(valid_barcodes), len(valid_barcodes)))*(lev_cutoff+1)
            
            for bcind, tbc in enumerate(tqdm(valid_barcodes.barcode, desc='Levenshtein dist. calc. progress')):
                ld_matrix[:bcind,bcind] = [levenshtein_distance(tbc, tvb, score_cutoff=lev_cutoff) for tvb in valid_barcodes.barcode[:bcind]]
            np.fill_diagonal(ld_matrix, (lev_cutoff+1))

            match_inds0 = np.where(ld_matrix <= lev_cutoff)
            valid_names = valid_barcodes.index

            match_pairs0 = []
            duplex_pairs = []
            
            # check for duplex reads and only append the ones to "match_pairs" that aren't
            for i,j in zip(match_inds0[0], match_inds0[1]):
                # if the matches came from separate channels in the flow cell, append them
                if valid_barcodes.loc[valid_names[i],'channel'] != valid_barcodes.loc[valid_names[j],'channel']:
                    match_pairs0.append((i,j))
                # if the matches are in the same channel but separated by more than 20 reads, append them
                elif np.abs(valid_barcodes.loc[valid_names[i],'read'] - valid_barcodes.loc[valid_names[j],'read']) > 20:
                    match_pairs0.append((i,j))
                # duplexes will have the opposite sense, if the sense of the matches is the same, append them
                elif ~(valid_barcodes.loc[valid_names[i],'reverse'] ^ valid_barcodes.loc[valid_names[j],'reverse']):
                    match_pairs0.append((i,j))
                else:
                    duplex_pairs.append((i,j))
            
            # convert good match pairs list into numpy array
            match_pairs0 = np.array(match_pairs0)
            duplex_pairs = np.array(duplex_pairs)
            print(duplex_pairs)
            # calculate the duplex number. That is the difference between the number repeated sequence before
            # and after filtering out duplexes

            # This code is an inelegant thing used to find barcodes that are marked as repeats due to a duplex.
            # e.g. suppose A and B are duplexes and C matches both. A and B are successfully removed by the duplex code,
            # however, A and C and B and C are both retained in match_pairs0 given an erroneous triple.
            # This code removes one of those.
            
            if len(duplex_pairs) > 0:
            
                dup_in_pairs = list(set(duplex_pairs[:,0]) & set(match_pairs0.flatten()))
                match_pairs = []
                for mp in match_pairs0:
                    keep = True
                    for d in dup_in_pairs:
                        if d in mp:
                            keep = False
                    if keep:
                        match_pairs.append(mp)
                match_pairs = np.array(match_pairs)
            else:
                match_pairs = match_pairs0
                dup_in_pairs = []


            duplex_number = duplex_pairs.shape[0]
            # get the total number of observed reads
            total_number = len(barcode_df)
            # get the number of good reads without a valid barcode
            no_barcode_number = total_number - len(valid_barcodes)
            # compute the number of single reads.
            single_number = total_number - no_barcode_number - 2*len(match_pairs) - (duplex_number - len(dup_in_pairs))
            # get the number of times barcodes are seen n-1 times.
            # repeat_counts[:,0] is the number of occurences a barcode - 1
            # repeat_counts[:,1] is the number of barcodes observed this many times.
            repeat_counts = np.unique(np.unique(match_pairs[:,0], return_counts=True)[1], return_counts=True)
            # create an array with the number of barcodes observed a different number of times
            # 0th position is the number of good reads without a barcode
            # 1st position is the number of barcodes observed once
            # 2nd position is the number of barcodes observed twice, and so on...
            repeats = np.zeros(np.max(repeat_counts[0])+ 2, int)
            repeats[0] = no_barcode_number
            repeats[1] = single_number
            for i,j in zip(repeat_counts[0], repeat_counts[1]):
                repeats[i+1] = j
            
            print("Characterizing library composition.")
            print(f"Total reads: {total_number}")
            print(f"Number of duplex reads: {duplex_number}")
            print(f"Number of singletons: {single_number}")
            try:
                print(f"Number of doubles: {repeats[2]}")
                try:
                    print(f"Number of three or more: {np.sum(repeats[3:])}")
                except:
                    print(f"No barcodes observed 3 or more times.")
            except:
                print("No barcodes observed 2 or more times.")


            return repeats, match_pairs, duplex_pairs
            
        print("")
        # compute the number of barcodes repeats
        self.repeats, self.match_pairs, self.duplex_pairs = find_recurring_barcodes(self.barcodes)

        if len(self.repeats) < 3:
            print("Can't compute library size estimate. Not enough repeated barcodes.")
            self.lib_size = {'const_prob_estimate': np.nan, 'preseq_prob_estimate': np.nan}
            return None
        

        
        const_prob_lib_est = self.repeats[1:].sum() / (2*(self.repeats[2]/self.repeats[1]))
        #lambda_est = np.array([(i+1)*j for i,j in enumerate(self.repeats)]).sum() / const_prob_lib_est

        print(f"Library estimate, assuming all barcodes equally likely: {int(const_prob_lib_est)}\n")
        self.lib_size = {'const_prob_estimate': const_prob_lib_est, 'preseq_prob_estimate': np.nan}
        
        # if using preseq
        # This code runs through a fairly convoluted logic chain and exits if problems are encountered at various junctures.
        if use_preseq:
            print("Attempting to use preseq.")

            # check that there are enough repeated observations for preseq to function. Here it needs to have a single barcode seen at least 6 times.
            if len(self.repeats) < 7:
                print("Not enough barcode repeats. Must have at least one barcode with 6 repeats to run preseq.")
                return
            
            # make temporary histogram filename            
            temp_hist = 'preseq.hist'

            # Write temp_hist with the histogram format that preseq is expecting.
            with open(temp_hist, 'w') as f:
                print("Writing temporary histogram file for preseq.")
                for ind, counts in enumerate(self.repeats[1:]):
                    f.write(f"{ind+1}\t{counts}\n")

            # temporary preseq output filename
            temp_out = 'preseq_temp.txt'
            # build the preseq command. This uses the population size function in histogram mode (-H)
            # The "tee" command here causes the output to both print to the terminal and save to the file temp_out
            preseq_command = f"preseq pop_size -H {temp_hist} | tee {temp_out}"
            # call the command
            preseq_err = os.system(preseq_command)

            # check if the command ran without an error (exited with 0)
            if preseq_err == 0:
                print("Preseq library size estimate exited succesfully. But may not have worked.\n")
                
                # Try to parse the preseq output by reading it into a pandas dataframe.
                try:
                    preseq_out = pd.read_csv(temp_out, delim_whitespace=True)
                    self.lib_size['preseq_prob_estimate'] = {'pop_size': preseq_out['pop_size_estimate'].to_numpy()[0],
                                                           'lower_ci': preseq_out['lower_ci'].to_numpy()[0],
                                                           'upper_ci': preseq_out['upper_ci'].to_numpy()[0]}
                    os.remove(temp_out)

                # if the output reading did not work print an error message and exit.    
                except:
                    print(f"There was some error parsing the preseq output. You can check {temp_out}\n")
                    return
                # If the parsing does work then update the lib_size dictionary with the preseq results including the size estimate and the confidence interval
                ests = [self.lib_size['preseq_prob_estimate'][k] for k in ['lower_ci', 'pop_size', 'upper_ci']]
                print(f"Preseq population size estimate: [ {ests[0]} --* {ests[1]} *-- {ests[2]} ]\n")
                return

            # runs if preseq exits abnormally
            else:
                print("There was an error running preseq.")
                return
        # if not running preseq just exit
        else:
            return


class MiserSeqs(BaseSeqs):
    """"

    Processing of MISER libraries with nanopore sequencing

    For initialization:
    MiserSeqs(ref_fasta, reads_fastq, casStart, casEnd)

    Parameters
    -----------------------------------------------------
    ref_fasta: string with name of full-length reference vector fasta
    reads_fastq: string with fastq file of nanopore reads
    casStart: integer, index of Cas9 start
    casEnd: integer, index of Cas9 end

    Functions
    -----------------------------------------------------
    fastq_info(): get some statistics and visualization of fastq
    align2ref(): align fastq sequences to the reference fasta
    find_barcodes(): find and extract barcodes from fastq sequences
    find_gaps(): find the location and size of deletions
    deletion_plot(): generate a scatterplot of identified deletions
    write_mapped_fastq(): writes a fastq of mapped deletion reads

    """
    def align2ref(self):
        """"

        Aligns fastq sequences to a reference plasmid fasta.

        Usage:
        align2ref()

        Description:
        This function aligns the fastq reads to a doubled version of the fasta vector
        sequence in order to deal with the wraparound circular DNA problem using
        minimap2.

        Parameters
        -----------------------------------------------------
        None

        Returns
        -----------------------------------------------------
        None
        But it creates a dataframe self.pairwise_dict that contains the nucleotide
        by nucleotide mapping.

        """
        super().align2ref(full_plasmid=True)
    
    
    def find_gaps(self, min_align_length = 1000):
        """"

        Locate deletions in library of MISER variants

        Usage:
        find_gaps(min_align_length = 1000)

        Description:
        The code analyzes the aligned pair indices from self.pairwise_dict and looks for the
        location at which the biggest index hop occurs in the query relative to the reference.
        It does this in a way that is somewhat insulated from sequencing noise by convolving
        the relative sequence registration by position with a step function. The location of
        the maximum absolute value of this quantity is taken as the gap. The gap size is
        computed by taking the median registration shift (could be positive for deletions
        or negative for insertions) from +/-10 positions from the gap location.

        Parameters
        -----------------------------------------------------
        min_align_length: integer, minimum length of alignment for which a gap will be computed

        Returns
        -----------------------------------------------------
        None
        Writes a dataframe self.gaps which contains the gap location and length for each read

        """

        if not hasattr(self, 'pairwise_dict'):
            raise ValueError("Must run align2ref method before find_gaps.")
        
        vec_length = len(self.ref_seqrecord.seq)

        # initial the gap dictionary
        info_dict = dict()

        # iterate over all of the entries in the pairwise dictionary
        for read_name in tqdm(list(self.pairwise_dict.keys()), desc="Progress"):
            # find the pairwise alignment that is the longest and get it's index
            largest_align = np.argmax([len(p['pairwise']) for p in self.pairwise_dict[read_name]])
            # grab the alignment array for this longest alignment
            pairwise = self.pairwise_dict[read_name][largest_align]['pairwise']
            # store the alignment length
            align_length = pairwise.shape[0]

            if align_length < min_align_length:
                continue

            # get the index (meaning position in alignment, not actual query index) for which the corresponding reference index is at a minimum
            # notice here that I compute a modulo with the vector length because the alignment was originally done a doubled version of the vector
            min_ref_ind = np.argmin(np.mod(pairwise[:,1], vec_length))
            # get the value of the reference alignment at this minimum.
            min_ref = np.mod(pairwise[min_ref_ind,1], vec_length)
            # get the index of the query at this position
            # (this is the value that would be used, for example, to permute fastq reads to be in the reference register)
            min_query_ind = pairwise[min_ref_ind, 0]

            # The following few steps are a little abstract but they are using a convolution to locate the biggest step in the
            # query-reference pairwise index alignments. (This is also, by far, the slowest part of this code and could probably be optimized.)
            # create a step function of twice the length of the alignment
            step = np.hstack((np.ones(align_length), -1*np.ones(align_length)))
            # convolve the query subtracted reference with the step to find the transition point
            convolve = np.convolve(pairwise[:,1]-pairwise[:,0] - np.mean(pairwise[:,1]-pairwise[:,0]), step, mode='valid')
            # get the location of the discontinuity as the max of the convolution
            del_ind = np.argmax(np.abs(convolve))
            
            # check to make sure that the discontinuity isn't too close to the edge where it will be very inaccurate
            # and can suffer from other artifacts.
            if (del_ind < 10) or (del_ind > len(pairwise)- 10):
                continue

            # compute the gap as the difference between the median relative index before and after the jump
            # sign will indicate deletion or insertion
            del_gap = np.median(pairwise[del_ind: del_ind + 10,1] - pairwise[del_ind: del_ind + 10,0]) - np.median(pairwise[del_ind - 10: del_ind,1] - pairwise[del_ind-10: del_ind,0])
            del_start_ind_ref = np.median(np.mod(pairwise[del_ind-10:del_ind+1,1], vec_length) + np.arange(10,-1,-1)).astype(int) 
            
            # build the dictionary entry
            info_dict[read_name] = {'del_start' : del_start_ind_ref,
                                    'del_gap' : del_gap,
                                'mapping_quality': self.pairwise_dict[read_name][largest_align]['mapping_quality'],
                                'mean_q': self.pairwise_dict[read_name][largest_align]['mean_q'],
                                'min_ref_ind': min_ref_ind,
                                'min_query_ind':min_query_ind}
        
        # turn the full dictionary into a pandas dataframe for ease of use
        self.gaps = pd.DataFrame.from_dict(info_dict, orient='index')

    def write_mapped_fastq(self, output_fastq_filename, min_gap = -np.inf):
        """"

        Write fastq file containing the mapped deletions.

        Usage:
        write_mapped_fastq(output_fastq_filename, min_gap = 0)

        Description:
        This code writes a fastq that contains the subset of sequences that were mapped to the Cas protein and exceed the minimum gap.

        Parameters
        -----------------------------------------------------
        output_fastq_filename: string, name of fastq file to be written. If no extension included with write with .fastq extension
        min_gap: int or float, minimum deletion size to include in export. Default is -inf so that all deletions and insertions are included.

        Returns
        -----------------------------------------------------
        None
        Writes output_fastq_filename to hard disk.

        """
        if not hasattr(self, 'gaps'):
            raise ValueError("Must run find_gaps method before exporting fastq.")       
        
        mapped_records = []
        for i in self.gaps[self.gaps['del_gap'] > min_gap].index:
            mapped_records.append(self.fastq_index[i])

        if '.f' not in output_fastq_filename:
            outfile = f'{output_fastq_filename}.fastq'
        else:
            outfile = output_fastq_filename
        
        with open(outfile,'w') as output_handle:
            SeqIO.write(mapped_records, output_handle, "fastq")

        print(f"Saved as: {outfile}")
        
        return None 


    # This creates a plot of the deletion points colored by the mapping quality score.
    # Can give optional arguments for filename to save with extension.
    # e.g. deletion_plot(df, outfilename0="test.png")
    def deletion_plot(self, outfilename0 = '', min_gap=10, cas_only=False):
        """"

        Graphical display of MISER library deletions

        Usage:
        deletion_plot(outfilename0 = '', min_gap=10, cas_only=False)

        Parameters
        -----------------------------------------------------
        outfilename0: if desired, provide an output image filename, e.g. outfilename0='my_image.png'
        min_gap: integer with the minimum gap to add to the plot, can also be negative to include insertions
        cas_only: boolean to restrict visualization window to Cas protein (defaults to False)

        Returns
        -----------------------------------------------------
        None
        Will write an output image if given a filename with extension in outfilename0

        """


        vec_length = len(self.ref_seqrecord.seq)
        df_good = self.gaps[self.gaps['del_gap']>=min_gap]
        plt.scatter(df_good['del_start'], df_good['del_start'] + df_good['del_gap'], s=0.5, c=df_good['mapping_quality'])
        plt.vlines(self.casStart, 0, vec_length, color='red', linestyles='dashed')
        plt.hlines(self.casEnd, 0 , vec_length, color='red', linestyles='dashed')
        plt.plot([0,vec_length],[0,vec_length],'k--')
        plt.axis('square')
        
        if cas_only:
            plt.xlim([self.casStart, self.casEnd])
            plt.ylim([self.casStart, self.casEnd])
        else:
            plt.xlim([0, vec_length])
            plt.ylim([0, vec_length])

        plt.xlabel('deletion start')
        plt.ylabel('deletion end')

        if outfilename0 != '':
            plt.savefig(outfilename0, dpi=300)
        plt.show()


class AmpliconSeqs(BaseSeqs):
    """"

    Processing of MISER library amplicons

    For initialization:
    AmpliconSeqs(ref_fasta, reads_fastq)

    Parameters
    -----------------------------------------------------
    ref_fasta: string with name of full-length reference vector fasta
    reads_fastq: string with fastq file of nanopore reads
    casStart: integer, index of Cas9 start
    casEnd: integer, index of Cas9 end

    Functions
    -----------------------------------------------------
    fastq_info(): get some statistics and visualization of fastq
    align2ref(): align fastq sequences to the reference fasta
    find_barcodes(): find and extract barcodes from fastq sequences

    """
    def __init__(self, ref_fasta, reads_fastq):
        self.ref_fasta = Path(ref_fasta)
        self.reads_fastq = Path(reads_fastq)
        self.fastq_index = SeqIO.index(str(self.reads_fastq), 'fastq')
        self.read_count = len(list(self.fastq_index.keys()))
        self.ref_seqrecord = SeqIO.read(str(self.ref_fasta), 'fasta')

        print(f'Indexing fastq: "{str(self.reads_fastq)}"')
        print(f'Fastq contains {self.read_count} reads.')
    def align2ref(self):
        """"

        Aligns fastq sequences to a reference fasta of the expected amplicon.

        Usage:
        align2ref()

        Description:
        This function aligns the fastq reads to a doubled version of the fasta vector
        sequence in order to deal with the wraparound circular DNA problem using
        minimap2.

        Parameters
        -----------------------------------------------------
        None

        Returns
        -----------------------------------------------------
        None
        But it creates a dataframe self.pairwise_dict that contains the nucleotide
        by nucleotide mapping.

        """
        super().align2ref(full_plasmid=False)

class DeletionVarCount:
    """"

    Class for counting the number of deletion variants in a fastq file

    For initialization:
    dvc = sequtils.DeletionVarCount(ref_fasta, reads_fastq)

    Parameters
    -----------------------------------------------------
    ref_fasta: string with name of full-length reference vector fasta
    reads_fastq: string with fastq file of nanopore reads

    Functions
    -----------------------------------------------------
    dvc.build_deletion_matrix(): build a deletion matrix from the reference fasta

    """
    def __init__(self, ref_fasta, reads_fastq):
        self.ref_fasta = Path(ref_fasta)
        self.reads_fastq = Path(reads_fastq)
        self.fastq_index = SeqIO.index(str(self.reads_fastq), 'fastq')
        self.read_count = len(list(self.fastq_index.keys()))
        
        print(f'Indexing fastq: "{str(self.reads_fastq)}"')
        print(f'Fastq contains {self.read_count} reads.')

        


    def build_deletion_matrix(self, viewplot=True):
        """"
        Build a deletion matrix from the reference fasta.

        Usage:
        lvc.build_deletion_matrix()

        """
        self.ref_fasta_sort = self.ref_fasta.with_name(f"{self.ref_fasta.stem}_sorted{self.ref_fasta.suffix}")
        self.wt_fasta = self.ref_fasta.with_name("wt_amplicon.fasta")

        # create a size-sorted fasta of variants
        var_lengths = []
        var_names = []
        for var_seq in SeqIO.parse(str(self.ref_fasta), 'fasta'):
            var_names.append(var_seq.id)
            var_lengths.append(len(var_seq.seq))
        self.var_name_sort = [var_names[i] for i in np.argsort(var_lengths)[::-1]]
        var_seqs_idx = SeqIO.index(str(self.ref_fasta), 'fasta')
        # Write sorted fasta (biggest to smallest)
        print(f"Writing sorted fasta of variants to {self.ref_fasta_sort}")
        SeqIO.write([var_seqs_idx[vs] for vs in self.var_name_sort], self.ref_fasta_sort, 'fasta')
        # Write wt-only fasta for minimap2 alignment.
        wtseq = var_seqs_idx[self.var_name_sort[0]]
        self.wt_length = len(wtseq.seq)
        print(f"Writing wt-only fasta to {self.wt_fasta} with length {self.wt_length}")
        SeqIO.write(wtseq, self.wt_fasta, 'fasta')

        #create alignment of variants.
        self.var_fasta_align = self.ref_fasta_sort.with_name(f"{self.ref_fasta_sort.stem}_aligned{self.ref_fasta_sort.suffix}")
        mafft_call = f"mafft {self.ref_fasta_sort} > {self.var_fasta_align}"
        print(f"Running mafft to create an MSA of deletion variants.")
        run_sys_command(mafft_call)

        # build deletion matrix
        var_alignment = AlignIO.read(self.var_fasta_align, 'fasta')
        del_mat = []
        for var_al in var_alignment:
            del_mat.append([ -1 if s == '-' else 1 for s in var_al.seq ])
        self.del_mat = np.array(del_mat)

        if viewplot:
            plt.figure(figsize=(12, 4))  # Adjust figure size

            # Force aspect ratio while preventing interpolation
            plt.imshow(del_mat, cmap='gray', aspect=(self.del_mat.shape[1]/self.del_mat.shape[0])*(1/3), interpolation='nearest')

            plt.title("Stacked deletion variants")
            plt.xlabel("position")
            plt.ylabel("variant")

            plt.show()
        return
    
    def assign_deletion_variants(self, threshold=0.90):
                                         
        """
        Assign deletion variants to the fastq reads.

        Usage:
        dvc.assign_deletion_variants()

        """

        # check that the deletion matrix has been built
        if not hasattr(self, 'del_mat'):
            print("Running build_deletion_matrix...")
            self.build_deletion_matrix(viewplot=False)

        print("Running minimap2 to align fastq reads to WT fasta.")
        # run minimap2 against only wt
        wt_align = self.wt_fasta.with_name("wt_align.sam")
        minimap2_call = f'minimap2 -ax map-ont {self.wt_fasta} {self.reads_fastq} > {wt_align}'
        run_sys_command(minimap2_call)

        print("Assigning deletion variants by scoring with the deletion matrix.")
        # initialize dictionary for variant assignments of reads
        read_assignment = {}
        # open minimap2 alignment and iterate over entries
        with pysam.AlignmentFile(wt_align, "r") as alignments:
            for alignment in alignments:
                # if an alignment is unmapped or not primary, move on
                if alignment.is_unmapped or alignment.is_secondary or alignment.is_supplementary:
                    continue

                # get the pairwise alignment vectors for the read
                pairwise = np.array(alignment.get_aligned_pairs())
                # only take the indices over which the reference is mapped
                valid_inds = np.where(pairwise[:,1] != None)[0]
                # create a vector that represents the query over valid indices as 1's for matches and -1's for deletions
                pairwise_match = [ -1 if p == None else 1 for p in pairwise[valid_inds,0]]
                # get the valid reference indices
                good_ref_inds = pairwise[valid_inds,1].astype(int)
                # initialize a deletion vector as all deletion
                del_vect = -1*np.ones(self.wt_length).astype(int)
                # fill it in for the mapped query overlap
                del_vect[good_ref_inds] = pairwise_match
                # compute an overalap score by matrix multiplication with the matrix of variant deletion fingerprints
                # the highest possible score is the length of the WT variant. (note that this highest score is possible
                # for any of the variants).
                overlap_score = np.matmul(self.del_mat, del_vect)
                # find the index of the highest score
                maxind = np.argmax(overlap_score)
                # get the value of the highest score
                maxval = overlap_score[maxind]
                # if the value of the highest is threshold or more of the maximum possible score, count it as a positive identification
                if maxval > threshold*self.wt_length:
                    read_assignment[alignment.query_name] = {'length': alignment.infer_read_length(),
                                                            'q-score': np.mean(alignment.query_qualities),
                                                            'assignment': self.var_name_sort[maxind]}
                
        self.df_assignments = pd.DataFrame.from_dict(read_assignment, orient='index')


        # count up the number of reads assigned to each variant
        var_counts_dict = {}
        for vns in self.var_name_sort:
            var_counts_dict[vns] = {'var_count': (self.df_assignments['assignment'] == vns).sum()}
        self.df_counts = pd.DataFrame.from_dict(var_counts_dict, orient='index')

        print("Assignments complete.")
        print(f"Found assignments for {len(self.df_assignments)}/{self.read_count} reads, {100*len(self.df_assignments)/self.read_count:.1f}%")
        print(f"Cleaning up temporary files: {wt_align}, {self.ref_fasta_sort}, {self.var_fasta_align}")
        os.remove(wt_align)
        os.remove(self.ref_fasta_sort)
        os.remove(self.var_fasta_align)
        print("")
        print("Finished.")

        return self.df_counts
    




