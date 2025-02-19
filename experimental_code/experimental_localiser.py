from psychopy import core, event, visual, monitors
import os
import pandas as pd
from const import (
    FULLSCREEN,
    START_KEY,
    TRIGGER_KEY,
    CYCLES_PER_DEGREE,
    OBJECTS_DIR,
    SCRAMBLED_DIR,
    OBJECT_SIZE,
    FIXATION_TIME,
    IMAGE_TYPE,
    STIM_ON_TIME,
    STIM_OFF_TIME,
    STIM_PRES,
    FLICKER_FREQUENCY,
    SCREEN,
    RET_BLOCK_TIME,
    MAX_RADIUS,
    THICKNESS,
    COOL_DOWN_TIME,
    TERMINATION_KEY,
    MONITOR_WIDTH,
    MONITOR_DISTANCE,
    MONITOR_RESOLUTION,
    FIXATION_SIZE,
    WARM_UP_TIME,
)


class Localizer:
    """
    Class to set up and execute differnet kinds of localizers and mischelaneous things in the experiment
    """

    def __init__(self, tracker, io):
        self.tracker = tracker
        self.io = io
        self.win = 0
        self.fixation = 0
        self.stimuli = 0
        self.stimulus = 0
        self.data = 0
        self.run_timer = core.Clock()
        self.block_timer = core.Clock()
        self.elapsed_time = 0
        self.radial_checkerboard = 0
        self.inner_mask = 0
        self.outer_radius = 0
        self.img = {}

    def _setup_hardware(self):
        monitor = monitors.Monitor("monitor", distance=MONITOR_DISTANCE, width=MONITOR_WIDTH)
        monitor.setSizePix(MONITOR_RESOLUTION)
        self.win = visual.Window(
            size=(MONITOR_RESOLUTION),
            units="pix",
            fullscr=FULLSCREEN,
            colorSpace="rgb",
            monitor=monitor, #"55w_60dist"
            color=[0, 0, 0],
            screen=SCREEN,
        )
        self.win.setMouseVisible(False)
        self.fixation = visual.Circle(
            self.win,
            radius=FIXATION_SIZE,
            fillColor="black",
            lineColor="black",
            edges=128,
        )
        self.data = pd.DataFrame(columns=['event', 'timestamp'])
        return

    def _prepare_stimuli(self):
        self.stimuli = {}
        self.stimuli["objects"] = []
        self.stimuli["scrambled"] = []
        object_files = os.listdir(OBJECTS_DIR)
        scrambled_files = os.listdir(SCRAMBLED_DIR)
        self.stimuli["objects"] = [
            visual.ImageStim(self.win, image=os.path.join(OBJECTS_DIR, file), size=OBJECT_SIZE)
            for file in object_files
            if file.endswith(".jpg")
        ]
        self.stimuli["scrambled"] = [
            visual.ImageStim(self.win, image=os.path.join(SCRAMBLED_DIR, file), size=OBJECT_SIZE)
            for file in scrambled_files
            if file.endswith(".jpg")
        ]
        self.img["objects"] = 0
        self.img["scrambled"] = 0
        return
    
    def retinotopic_localizer(self):
        self._setup_hardware()
        self.show_text("Retinotopic Localizer")
        event.waitKeys(keyList=[START_KEY])
        self.wait_for_fMRI_trigger()
        self.run_timer.reset()
        core.wait(WARM_UP_TIME) #To give scanner time to warm up
        self.fixation.draw()
        self.win.flip()

        self.save_data("Run Start")
        for i in range(6):
            if i == 3:
                self.fixation.draw()
                self.win.flip()
                core.wait(5) #To give scanner time to catch up between normal and reverse retinotopy
            self.block_end = False
            self.save_data(f"Block{i+1} Start")
            self.block_timer.reset()
            while not self.block_end:
                self.elapsed_time = self.block_timer.getTime()
                self.set_ret_stimulus(i=i)
                self.flicker_ret_stimulus()
                self.check_terminate_experiment()
            
        self.save_data("Run End")
        self.fixation.draw()
        self.win.flip()
        core.wait(COOL_DOWN_TIME) #To give Scanner time to catch up
        self.win.close()
        return self.data
    
    def set_ret_stimulus(self, i):
        if i < 3:
            self.outer_radius = (self.elapsed_time / RET_BLOCK_TIME) * MAX_RADIUS
            inner_radius = (self.outer_radius - THICKNESS)
        else:
            self.outer_radius = MAX_RADIUS - (self.elapsed_time / RET_BLOCK_TIME) * MAX_RADIUS
            self.outer_radius = max(0, self.outer_radius)  # Ensure outer_radius doesn't go below 0
            inner_radius = self.outer_radius - THICKNESS
        radial_cycles = round(CYCLES_PER_DEGREE * 2 * self.outer_radius)
        self.radial_checkerboard = visual.RadialStim(
            self.win,
            tex="sqrXsqr",
            size=self.outer_radius * 2,
            pos=(0, 0),
            radialCycles=radial_cycles,
            angularCycles=32,
            visibleWedge=(0, 360),
        )
        self.inner_mask = visual.Circle(
            self.win,
            radius=inner_radius,
            edges=128,
            fillColor="grey",
            lineColor="grey",
            pos=(0, 0),
        )
        return
    
    def flicker_ret_stimulus(self):
        if self.elapsed_time <= RET_BLOCK_TIME:
            self.block_end = False
        else:
            self.block_end = True
            return
            
        if core.getTime() % (1 / FLICKER_FREQUENCY) < (1 / FLICKER_FREQUENCY / 2):
                self.radial_checkerboard.contrast = 1
        else:
            self.radial_checkerboard.contrast = -1

        self.radial_checkerboard.draw()
        if self.outer_radius > THICKNESS:
            self.inner_mask.draw()
            self.fixation.draw()
        self.win.flip()
   
    def object_localizer(self):
        self._setup_hardware()
        self._prepare_stimuli()
        self.show_text("Object Localizer")
        event.waitKeys(keyList=[START_KEY])
        self.wait_for_fMRI_trigger()
        self.run_timer.reset()
        core.wait(WARM_UP_TIME) #To give scanner time to warm up

        for i in range(6):
            self.block_timer.reset()
            self.fixation.draw()
            self.win.flip()
            self.save_data(f"fixation_B{i+1}")
            time = FIXATION_TIME
            while self.block_timer.getTime() < time:
                pass

            for t in range(len(IMAGE_TYPE)):
                type = IMAGE_TYPE[t]
                self.save_data(f"{type}_B{i+1}_S{t+1}")

                for j in range(20):
                    stim = STIM_PRES[self.img[type]]
                    object_image = self.stimuli[type][stim]
                    object_image.draw()
                    self.fixation.draw()
                    self.win.flip()
                    time = time + STIM_ON_TIME
                    while self.block_timer.getTime() < time:
                        pass

                    self.fixation.draw()
                    self.win.flip()
                    time = time + STIM_OFF_TIME
                    self.img[type] += 1
                    while self.block_timer.getTime() < time:
                        pass
                    self.check_terminate_experiment()

            IMAGE_TYPE.reverse()

        self.save_data("Run End")
        self.fixation.draw()
        self.win.flip()
        core.wait(COOL_DOWN_TIME) # To give Scanner time to catch up
        self.win.close()
        return self.data

    def wait_for_fMRI_trigger(self):
        #TODO Add reset reun_timer to this method (same for experimental_condition.py)
        """
        Waits for input from fMRI to start the experimental run
        """
        print("Waiting for scanner trigger...")
        event.waitKeys(keyList=[TRIGGER_KEY])
        print("Trigger received. Starting run")

    def show_text(self, text):
        presented_text = visual.TextStim(self.win, text=text, pos=[0, 0], height=60)
        presented_text.draw()
        self.win.flip()
        
    def save_data(self, marker=str):
        """
        Saves, prints and sends timestamps to eye tracker
        """
        current_time = self.run_timer.getTime()
        self.data.loc[len(self.data)] = [marker, current_time]
        if self.tracker:
            self.io.sendMessageEvent(text=marker, sec_time=current_time)
        return
    
    def check_terminate_experiment(self):
        keys = event.getKeys()
        if TERMINATION_KEY in keys:
            print("Experiment terminated")
            core.quit()