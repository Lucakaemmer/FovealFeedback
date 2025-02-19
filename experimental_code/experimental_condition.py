from experimental_run import ExperimentalRun
from psychopy import visual, event, core, monitors
import numpy as np
from const import (
    STIMULUS_COORDINATES,
    FOVEAL_COORDINATES,
    TRIGGER_KEY,
    START_KEY,
    CONDITION_TEXT,
    CONTROL_TEXT,
    FULLSCREEN,
    STIMULUS_SIZE,
    STIMULUS_DIR,
    SCREEN,
    COOL_DOWN_TIME,
    MONITOR_DISTANCE,
    MONITOR_WIDTH,
    MONITOR_RESOLUTION,
    WARM_UP_TIME,
)


class ExperimentalCondition:
    """
    A class to set up and execute different conditions of a the experiment.

    Attributes:
        stimulus_positions (list): Positions where stimuli will be displayed.
        position (str): Determines if stimuli presented in 'fovea' or 'periphery'.
        eye_trigger (bool): Flag to determine if stimuli disappear when looked at.
        eyelink (pylink.EyeLink): EyeLink object for eye-tracking data collection.

    Methods:
        _get_stimulus_positions(position): Determines the stimulus positions based on the specified position category.
        execute_run(stimulus, mouse_stimulation=False): Executes the experimental run with the given stimulus.
    """

    def __init__(self, position, eye_trigger, tracker, io):
        """
        Parameters:
            position (str): Determines if stimuli presented in 'fovea' or 'periphery'.
            eye_trigger (bool): Flag to determine if stimuli disappear when looked at.
            eyelink (pylink.EyeLink): EyeLink object for eye-tracking data collection.
        """
        self.stimulus_positions = self._get_stimulus_positions(position)
        self.position = position
        self.eye_trigger = eye_trigger
        self.tracker = tracker
        self.io = io
        self.win = 0
        self.mouse = 0
        self.fixation = 0

    def _get_stimulus_positions(self, position):
        """
        Determines the stimulus positions based on the specified position category.

        Parameters:
            position (): Determines if stimuli presented in 'fovea' or 'periphery'
        """
        if position == "periphery":
            stim_positions = STIMULUS_COORDINATES
        elif position == "fovea":
            stim_positions = FOVEAL_COORDINATES
        else:
            raise ValueError(
                'Invalid position. Please choose between "fovea" or "periphery".'
            )
        return stim_positions

    def execute_run(self, mouse_stimulation=False, n_run=int):
        """
        Executes the experimental run with the given stimulus.

        Parameters:
            stimulus (): Determines which stimulus should be used in a given run.
            mouse_stimulation (): Determines if mouse movement should substitute eye tracking
            window (): Defines the dimensions and settings of the presentation window
        """
        self.setup_hardware()
        stimuli = self.prepare_stimuli()
        run = ExperimentalRun(
            stimulus_positions=self.stimulus_positions,
            eye_trigger=self.eye_trigger,
            tracker=self.tracker,
            io=self.io,
            position=self.position,
            mouse_stimulation=mouse_stimulation,
            window=self.win,
            mouse=self.mouse,
            stimuli=stimuli,
            n_run=n_run
        )
        
        #self.check_flip_speed()
        
        run_timer = core.Clock()
        self.show_text()
        event.waitKeys(keyList=[START_KEY])
        self.wait_for_fMRI_trigger()
        run_timer.reset()
        core.wait(WARM_UP_TIME) #To give scanner time to warm up

        data, gaze_distances = run.loop_blocks(run_timer=run_timer)

        core.wait(COOL_DOWN_TIME)  #To give the scanner time to catch up with the recording
        self.win.close()
        return data, gaze_distances

    def setup_hardware(self):
        monitor = monitors.Monitor("monitor", distance=MONITOR_DISTANCE, width=MONITOR_WIDTH)
        monitor.setSizePix(MONITOR_RESOLUTION)
        self.win = visual.Window(
            size=(MONITOR_RESOLUTION),
            units="pix",
            screen = SCREEN,
            fullscr=FULLSCREEN,
            colorSpace="rgb",
            monitor="55w_60dist",
            color=[0, 0, 0],
        )
        if self.tracker !=0:
            self.win.setMouseVisible(False)
        self.mouse = event.Mouse(win=self.win)
        return

    def prepare_stimuli(self):
        stimuli = {}
        stimuli["horn"] = visual.ImageStim(
            self.win, image=STIMULUS_DIR["horn"], size=STIMULUS_SIZE
        )
        stimuli["tiger"] = visual.ImageStim(
            self.win, image=STIMULUS_DIR["tiger"], size=STIMULUS_SIZE
        )
        stimuli["guitar"] = visual.ImageStim(
            self.win, image=STIMULUS_DIR["guitar"], size=STIMULUS_SIZE
        )
        stimuli["kangaroo"] = visual.ImageStim(
            self.win, image=STIMULUS_DIR["kangaroo"], size=STIMULUS_SIZE
        )
        
        self.fixation = visual.Circle(self.win, radius=0.05, edges=128)
        return stimuli

    def wait_for_fMRI_trigger(self):
        """
        Waits for input from fMRI to start the experimental run
        """
        print("Waiting for scanner trigger...")
        event.waitKeys(keyList=[TRIGGER_KEY])
        print("Trigger received. Starting run...")

    def show_text(self):
        if self.eye_trigger:
            text = CONDITION_TEXT
        else:
            text = CONTROL_TEXT
        presented_text = visual.TextStim(self.win, text=text, pos=[0, 0], height=60)
        presented_text.draw()
        self.win.flip()

    def check_flip_speed(self):
        flip_times = []
        flip_timer = core.Clock()
        for i in range(1000):
            self.fixation.draw()
            flip_timer.reset()
            self.win.flip()
            flip_time = flip_timer.getTime()
            flip_times.append(flip_time)
        print(f"mean_flip_time: {np.mean(flip_times)}")
        print(f"max_flip_time: {np.max(flip_times)}")
        print(f"min_flip_time: {np.min(flip_times)}")
        return