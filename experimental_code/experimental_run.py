from psychopy import visual, core, event
import pickle
import os
import math
import random
import numpy as np
import pandas as pd
from const import (
    STIMULUS_SEQUENCES,
    N_BLOCKS,
    DISTANCE_FROM_STIMULUS,
    TIME_BLOCK,
    TIME_WAIT,
    STIMULUS_LIST,
    TRIGGER_KEY,
    ITI,
    TERMINATION_KEY,
    FIXATION_SIZE,
    FIXATION_DISTANCE,
)


class ExperimentalRun:
    """
    Defines the blocks that are executed in the experiemnt and how they are looped within a block

    Attributes:
        stimulus (str): Path to stimulus image file.
        self.stimulus_name (str): Name of the stimulus used in a given run.
        stimulus_positions (list): Positions where stimuli will be displayed.
        position (str): Determines if stimuli presented in 'fovea' or 'periphery'.
        eye_trigger (bool): Flag to determine if stimuli disappear when looked at.
        mouse_stimulation (bool): Flag to determine if mouse should be used instead of eye tracking.
        tracker (pylink.EyeLink): EyeLink object for eye-tracking data collection.

    Methods:
        loop_blocks(): Loops over experimental bocks.
        block(block_n): Executes a single block of the experiment.
        display_fixation_points(): Displays fixation points on the screen.
        display_image(stim_pos): Displays the stimulus image at a specified position.
        check_stimulus_distance(stim_pos): Breaks when gaze comes close to stimulus
        def get_gaze_position(self): Check the current gaze or mouse position depending if debugging mode is turned on
    """

    def __init__(
        self,
        stimulus_positions,
        position,
        eye_trigger,
        mouse_stimulation,
        tracker,
        io,
        window,
        mouse,
        stimuli,
        n_run,
    ):
        """
        Parameters:
            stimulus (str): Path to stimulus image file
            stimulus_positions (list): Positions where stimuli will be displayed
            position (str): Determines if stimuli presented in fovea or periphery
            eye_trigger (bool): Flag to determine if stimuli disappear when looked at
            mouse_stimulation (bool): Flag to determine if mouse should be used instead of eye tracking
            tracker (pylink.EyeLink): EyeLink object for eye-tracking data collection.
        """
        self.stimulus_positions = stimulus_positions
        self.eye_trigger = eye_trigger
        self.position = position
        self.mouse_stimulation = mouse_stimulation
        self.tracker = tracker
        self.io = io
        self.win = window
        self.mouse = mouse
        self.fixation = visual.Circle(self.win, radius=FIXATION_SIZE, edges=128)
        self.stimuli = stimuli
        self.n_run = n_run
        self.stimulus = 0
        self.stimulus_name = 0
        self.stim_sequence = 0
        self.run_timer = 0
        self.block_timer = core.Clock()
        self.data = 0
        self.gaze_distance_fixation = 0
        self.gaze_distance_trigger = 0
        self.gaze_distance_post_flip = 0

    def _initiate_data_containers(self):
        self.data = pd.DataFrame(columns=['event', 'timestamp', 'gaze'])
        self.gaze_distance_fixation = []
        self.gaze_distance_trigger = []
        self.gaze_distance_post_flip = []

    def loop_blocks(self, run_timer):
        """
        Loops over the experimental blocks within a run
        """
        self._initiate_data_containers()
        self.run_timer = run_timer
        self.save_data(f"Run {self.n_run} Start")
        block_counter = 0
        stim_sequences = STIMULUS_SEQUENCES.copy()
        random.shuffle(stim_sequences)
        for i in range(N_BLOCKS):
            block_counter = block_counter + 1
            self.stimulus_name = STIMULUS_LIST[i]
            self.stimulus = self.stimuli[self.stimulus_name]
            self.stim_sequence = stim_sequences[i]
            if self.eye_trigger:
                self.experimental_block(block_n=block_counter)
            else:
                self.control_block(block_n=block_counter)
            self.check_terminate_experiment()
        self.save_data(f"Run {self.n_run} End")
        gaze_distances = self.save_fixation_data()
        return self.data, gaze_distances

    def experimental_block(self, block_n):
        """
        Executes the experimental block of the experiment (condition 1). In this condition the stimulus presentation times are saved to be reused in condition 2 and 3.

        Parameters:
            block_n (int): Gives the current block within this run that is executed at the moment
        """
        self.block_timer.reset()
        self.display_fixation_point(fix_pos=self.stimulus_positions[self.stim_sequence[0]], color="white")
        self.win.flip()
        self.save_data(f"Block {block_n} Start")
        presentation_times = []
        fixation_times = []
        if not os.path.exists("presentation_times"):
            os.makedirs("presentation_times")
        if not os.path.exists("fixation_times"):
            os.makedirs("fixation_times")
        while self.block_timer.getTime() < TIME_WAIT:
            self.check_terminate_experiment()
            pass
        distance_fix = self.get_gaze_distance(stim_pos=self.stimulus_positions[self.stim_sequence[0]])
        self.gaze_distance_fixation.append(distance_fix)
        print(distance_fix)
        self.block_timer.reset()
        i = 1
        while True:
            old_fix_pos = self.stimulus_positions[self.stim_sequence[i-1]]
            stim_pos = self.stimulus_positions[self.stim_sequence[i]]
            self.save_data(f"Block {block_n} Stimulus {i}")
            self.display_image(stim_pos=stim_pos, old_fix_pos=old_fix_pos)
            self.display_fixation_point(fix_pos=stim_pos, color="black")
            self.check_stimulus_distance(stim_pos=stim_pos)
            self.win.flip()
            
            self.gaze_distance_post_flip.append(self.get_gaze_distance(stim_pos=stim_pos))
            presentation_times.append(self.block_timer.getTime())
            self.save_data(f"Block {block_n} Stimulus {i} caught")
            
            self.wait_ITI()
            self.wait_for_fixation(fix_pos=stim_pos)
            fixation_times.append(self.block_timer.getTime())
            
            if self.block_timer.getTime() > TIME_BLOCK:
                break
            self.gaze_distance_fixation.append(self.get_gaze_distance(stim_pos=stim_pos))
            i += 1
        self.save_data(f"Block {block_n} End")
        with open(f"presentation_times/block_{block_n}_{self.stimulus_name}.pkl", "wb") as file:
            pickle.dump(presentation_times, file)
        with open(f"fixation_times/block_{block_n}_{self.stimulus_name}.pkl", "wb") as file:
            pickle.dump(fixation_times, file)
        return

    def control_block(self, block_n):
        """
        Executes the control blocks of the experiment (condition 2 & 3). In this condition the presentation times recorded in condition 1 are loaded
        to match the timing of stimulus presentation to that of condition 1.

        Parameters:
            block_n (int): Gives the current block within this run that is executed at the moment
        """
        self.block_timer.reset()
        self.display_fixation_point(fix_pos=self.stimulus_positions[self.stim_sequence[0]], color="white")
        self.win.flip()
        self.save_data(f"Block {block_n} Start")
        with open(f"presentation_times/block_{block_n}_{self.stimulus_name}.pkl", "rb") as f:
            presentation_times = pickle.load(f)
        with open(f"fixation_times/block_{block_n}_{self.stimulus_name}.pkl", "rb") as f:
            fixation_times = pickle.load(f)
        while self.block_timer.getTime() < TIME_WAIT:
            pass
        self.block_timer.reset()

        for i in range(len(presentation_times)):
            stim_pos = self.stimulus_positions[self.stim_sequence[i + 1]]
            self.display_image(stim_pos=stim_pos)
            self.gaze_distance_fixation.append(self.get_gaze_distance(stim_pos=stim_pos))
            while self.block_timer.getTime() < presentation_times[i]:
                pass
            self.save_data(f"Block {block_n} Stimulus {i+1}")
            self.display_fixation_point(fix_pos=stim_pos, color="black")
            self.win.flip()
            while self.block_timer.getTime() < fixation_times[i]:
                pass

        self.save_data(f"Block {block_n} End")
        return

    def display_fixation_point(self, fix_pos, color):
        """
        Defines where fixation points are presented and what they look like.

        Parameter:
            middle_color (): Defines the color of the central fixation point
        """
        self.fixation.pos = fix_pos
        self.fixation.color = color
        self.fixation.draw()

    def display_image(self, stim_pos, old_fix_pos=0):
        """
        Defines which image is presented and where is is presented

        Parameters:
            stim_pos (): Coordinates of the stimulus that should be presented
        """
        self.stimulus.setPos(stim_pos)
        self.stimulus.draw()
        self.fixation.pos = stim_pos
        self.fixation.color = "black"
        self.fixation.draw()
        if self.eye_trigger:
            self.fixation.pos = old_fix_pos
            self.fixation.color = "black"
            self.fixation.draw()
        self.win.flip()

    def check_stimulus_distance(self, stim_pos):
        """
        Checks how far the gaze is away from a stimulus and breaks when gaze comes too close.
        Takes into account the edge case that tracker doesn"t return gaze.

        Parameters:
            stim_pos (): Coordinates of the presented stimulus to check its distance from gaze.
        """
        while True:
            gaze_distance = self.get_gaze_distance(stim_pos)
            if gaze_distance < DISTANCE_FROM_STIMULUS:
                self.gaze_distance_trigger.append(gaze_distance)
                break
                
            if self.block_timer.getTime() > TIME_BLOCK:
                self.gaze_distance_trigger.append(float('nan'))
                break

    def get_gaze_distance(self, stim_pos):
        """
        Gets the distance between the current gaze and the position of the stimulus
        
        Parameters:
            stim_pos (): Coordinates of the presented stimulus to check its distance from gaze.
        """
        gaze_position = self.tracker.getLastGazePosition()
        valid_gaze_pos = isinstance(gaze_position, (tuple, list))
        if valid_gaze_pos:
            gaze_distance = math.sqrt((gaze_position[0] - stim_pos[0])**2 + (gaze_position[1] - stim_pos[1])**2)
        else: 
            gaze_distance = 9999
        return gaze_distance

    def wait_for_fMRI_trigger(self):
        """
        Waits for input from fMRI to start the experimental run
        """
        print("Waiting for scanner trigger...")
        event.waitKeys(keyList=[TRIGGER_KEY])
        print("Trigger received. Starting run")

    def save_data(self, marker=str):
        """
        Saves, prints and sends timestamps to eye tracker
        """
        current_time = self.run_timer.getTime()
        current_gaze = self.tracker.getLastGazePosition()
        self.data.loc[len(self.data)] = [marker, current_time, current_gaze]
        if not self.mouse_stimulation:
            self.io.sendMessageEvent(text=marker, sec_time=current_time)
        return

    def save_fixation_data(self):
        """
        Gathers the distance data that was collected during the run and saves it in a dataframe
        """
        if self.eye_trigger:
            gaze_distances = pd.DataFrame(columns=['fixation', 'trigger', 'post_flip'])
            gaze_distances['fixation'] = self.gaze_distance_fixation
            gaze_distances['trigger'] = self.gaze_distance_trigger
            gaze_distances['post_flip'] = self.gaze_distance_post_flip
            print(f"mean fixation distance: {np.mean(self.gaze_distance_fixation)}")
            print(f"mean trigger distance: {np.mean(self.gaze_distance_trigger)}")
            print(f"mean post_flip distance: {np.mean(self.gaze_distance_post_flip)}")
        else:
            gaze_distances = pd.DataFrame(columns=['fixation'])
            gaze_distances['fixation'] = self.gaze_distance_fixation
            print(f"mean fixation distance: {np.mean(self.gaze_distance_fixation)}")
        return gaze_distances

    def wait_ITI(self):
        """
        Checks when 11 seconds of Block have elapsed and triggers inter-trial-interval of up to 0.5±0.2  seconds
        """
        remaining_block_time = TIME_BLOCK - self.block_timer.getTime()
        wait_time = min(ITI, max(0, remaining_block_time))
        core.wait(wait_time)

    def wait_for_fixation(self, fix_pos):
        """
        Waits until participant fixates on the fixation point before resuming the experiment. Resumes either if fixation is detected or if block time is exceeded.
        
        Parameters:
            stim_pos (): Coordinates of the presented stimulus to check its distance from gaze.
        """
        while True:
            gaze_distance = self.get_gaze_distance(fix_pos)
            if gaze_distance < FIXATION_DISTANCE:
                break
            if self.block_timer.getTime() > TIME_BLOCK:
                break

    def check_terminate_experiment(self):
        """
        Checks whether the termination key has been pressed 
        """
        keys = event.getKeys()
        if TERMINATION_KEY in keys:
            print("Experiment terminated")
            core.quit()