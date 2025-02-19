"""
Gaze analysis routines to process and plot eye-tracking data.
"""

import os
import glob
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from utils.const import SUBJECTS, LABEL_NAMES, LABELS, STIMULUS_SEQUENCES

class GazeAnalysis:
    """
    Class for performing gaze analysis.
    """

    def __init__(self):
        """Initialize with default settings and empty dataframes."""
        self.subjects = SUBJECTS
        self.subject = 0
        self.eye_dir = 0
        self.data = 0
        self.gaze = 0
        self.gaze_data = pd.DataFrame(columns=['subject', 'run', 'block', 'stimulus', 'saccades', 'min_distance', 'max_distance', 'mean_distance', 'stim_duration'])
        self.compare_distance = pd.DataFrame(columns=['subject', 'online', 'offline'])
        
    def run_gaze_analysis(self, measure):
        """Run gaze data analysis for all subjects using the given measure."""
        for s in range(len(self.subjects)):
            self.subject = self.subjects[s]
            self.eye_dir = f"/BULK/lkaemmer/data/foveal_decoding/data_eye/{self.subject}"
            self.import_data()
            self.extract_data(measure)
                
    def import_data(self):
        """Import gaze and data CSV files for the current subject."""
        # Get sorted file paths for both gaze and data files in condition 1
        data_file_pattern = os.path.join(self.eye_dir, '*data*C1*.csv')
        gaze_file_pattern = os.path.join(self.eye_dir, '*gaze*C1*.csv')
        data_file_paths = sorted(glob.glob(data_file_pattern), key=lambda x: int(os.path.basename(x).split('_R')[1].split('.')[0]))
        gaze_file_paths = sorted(glob.glob(gaze_file_pattern), key=lambda x: int(os.path.basename(x).split('_R')[1].split('.')[0]))
        
        # Get both gaze and data file for each run
        self.data = {}
        self.gaze = {}
        for f in range(len(data_file_paths)):
            data_name = os.path.basename(data_file_paths[f]).split('_')[-1].split('.')[0]
            gaze_name = os.path.basename(gaze_file_paths[f]).split('_')[-1].split('.')[0]
            
            self.data[data_name] = pd.read_csv(data_file_paths[f])
            self.gaze[gaze_name] = pd.read_csv(gaze_file_paths[f])
            self.gaze[gaze_name].replace(9999, np.nan, inplace=True)
            self.gaze[gaze_name] /= 88.7
            
    def extract_data(self, measure):
        """Extract relevant data from the imported files."""
        for run in self.data.keys():
            # Count the number of saccades in each block for all runs
            saccade_count_run = []
            for block_num in range(1, 21):
                # Count the number of saccades for each block
                count = self.data[run]['event'].str.contains(f"Block {block_num} .*caught", na=False).sum()
                saccade_count_run.append(count)
                
            # Creating block indeces for the gaze data and add it to the dataframe
            block_indices = []
            for i in range(len(saccade_count_run)):
                block_indices.extend([i+1] * saccade_count_run[i])
            self.gaze[run]["Block"] = block_indices    
            
            # Get minimum gaze distance at stimulus disappearance for each block and create exclude file
            min_distance = self.gaze[run].groupby('Block')[measure].min()
            max_distance = self.gaze[run].groupby('Block')[measure].max()
            mean_distance = self.gaze[run].groupby('Block')[measure].mean()
            all_distances = self.gaze[run].groupby('Block')[measure].apply(list).tolist()
            
            # Get Presentation Times for each Block
            duration = []
            for block_num in range(1, 21):
                block_data = self.data[run][self.data[run]['event'].str.startswith(f'Block {block_num} Stimulus')]
                timestamps = block_data['timestamp'].to_numpy()
                start_time = timestamps[::2]
                end_time = timestamps[1::2]
                duration.append(sum(end_time - start_time))
            
            data = pd.DataFrame({'subject': self.subject, 'run':run, 'block': list(range(1, 21)), 'stimulus': LABELS, 'saccades':saccade_count_run, 
                                         'min_distance': min_distance, 'max_distance': max_distance, 'mean_distance': mean_distance, 'all_distances': all_distances, 'stim_duration': duration})   
            
            if self.gaze_data.empty:
                self.gaze_data = data
            else:
                self.gaze_data = pd.concat([self.gaze_data, data], ignore_index=True)
                
        
            
    def exclude_data(self, saccade_min, distance_min):
        """Exclude blocks based on saccade count and minimum distance criteria."""
        block_count = len(self.gaze_data)
        self.gaze_data = self.gaze_data[self.gaze_data['saccades'] >= saccade_min]
        block_count_saccade = len(self.gaze_data)
        self.gaze_data = self.gaze_data[self.gaze_data['min_distance'] >= distance_min]
        block_count_distance = len(self.gaze_data)
        print(f"Saccade Count Exclude: {block_count - block_count_saccade} Blocks")
        print(f"Distance Exclude:      {block_count_saccade - block_count_distance} Blocks")
        print(f"Percent Excluded:      {(block_count_saccade - block_count_distance) / block_count * 100 }%")
        print(f"Remaining Blocks:      {block_count_distance} Blocks")

        
    def plot_gaze_distance(self, measure, distance, max):
        """Plot histogram of gaze distances from stimulus."""
        all_gaze_data = self.get_gaze_distances(exclude=measure, distance=distance)
        
        plt.figure(figsize=(10, 6))
        
        if measure == 'online':
            flattened_values = all_gaze_data['all_distances']
            flattened_values_edge = [value - 1.5 for value in flattened_values]
        elif measure == 'offline':
            flattened_values = all_gaze_data['offline_distances']
            flattened_values_edge = [value - 1.5 for value in flattened_values]

        sns.histplot(flattened_values_edge, bins=500, kde=True, alpha=0.9, color='#B0A7AB', label='Stimulus Edge', zorder=1, edgecolor='none')
        sns.histplot(flattened_values, bins=500, kde=True, alpha=0.9, color='#320A17', label='Stimulus Center', zorder=2, edgecolor='none')
        
        plt.axvspan(0, 1, color='grey', alpha=0.2)
        plt.xlim(0, 8) 
        plt.ylim(0, max) 
        # plt.title('Gaze Distance from Stimulus at Disappearance')
        # plt.xlabel('Distance in Visual Degrees')
        # plt.ylabel('Trials')
        plt.legend()
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        plt.show()

    def get_gaze_distances(self, exclude, distance):
        """Get gaze distances, optionally excluding blocks based on distance criteria."""
        all_gaze_data = self.gaze_data.explode('all_distances')
        all_gaze_data['all_distances'] = all_gaze_data['all_distances'].astype(float)

        offline_distances = []
        for subject in self.subjects:
            path = f'/BULK/lkaemmer/data/foveal_decoding/data_eye/participant_data/{subject}.rea'
            df_file = pd.read_csv(path, delimiter='\t', header=None)
            distances = df_file.iloc[:, 14]
            offline_distances.extend(distances)
        all_gaze_data['offline_distances'] = offline_distances
        
        if exclude == 'online':
            min_dist_per_block = all_gaze_data.groupby(['subject', 'run', 'block'])['all_distances'].min().reset_index()
            blocks_to_keep = min_dist_per_block[min_dist_per_block['all_distances'] >= distance][['subject', 'run', 'block']]
            all_gaze_data = all_gaze_data.merge(blocks_to_keep, on=['subject', 'run', 'block'], how='inner')

        elif exclude == 'offline':
            min_dist_per_block = all_gaze_data.groupby(['subject', 'run', 'block'])['offline_distances'].min().reset_index()
            blocks_to_keep = min_dist_per_block[min_dist_per_block['offline_distances'] >= distance][['subject', 'run', 'block']]
            all_gaze_data = all_gaze_data.merge(blocks_to_keep, on=['subject', 'run', 'block'], how='inner')
        
        return all_gaze_data

