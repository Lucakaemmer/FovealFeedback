from psychopy import core, visual, monitors, iohub
from psychopy.iohub import launchHubServer
from psychopy.iohub.util import hideWindow, showWindow
from psychopy.iohub.client.eyetracker.validation import TargetStim
from const import FULLSCREEN, SCREEN, MONITOR_RESOLUTION, MONITOR_DISTANCE, MONITOR_WIDTH

class EyeTracker:
    """
    A class that initiates the eye tracker, performs calibration, validation and drift correction and handles the data
    """

    def __init__(self, session_id):
        """ """
        self.session_id = session_id
        self.tracker = 0
        self.keyboard = 0
        self.io = 0
        self.win = 0
        self.text_stim = 0
        self.target_stim = 0
        self.eyetracker_config = self._get_eyetracker_config()
        self.devices_config = self._get_devices_config()
        self.validation_proc = 0

    def _get_eyetracker_config(self):
        eyetracker_config = dict(name="tracker")
        eyetracker_config["model_name"] = "EYELINK 1000 DESKTOP"
        eyetracker_config["simulation_mode"] = False
        eyetracker_config["runtime_settings"] = dict(
            sampling_rate=1000, track_eyes="RIGHT"
        )
        eyetracker_config["calibration"] = dict(
            auto_pace=True,
            target_duration=1.5,
            target_delay=1.0,
            screen_background_color=(0, 0, 0),
            type="THIRTEEN_POINTS",
            unit_type="pix",
            color_type=None,
            target_attributes=dict(
                outer_diameter=25,
                inner_diameter=9,
                outer_fill_color=[-0.5, -0.5, -0.5],
                inner_fill_color=[-1, 1, -1],
                outer_line_color=[1, 1, 1],
                inner_line_color=[-1, -1, -1],
            ),
        )
        return eyetracker_config

    def _get_devices_config(self):
        devices_config = {}
        devices_config["eyetracker.hw.sr_research.eyelink.EyeTracker"] = self.eyetracker_config
        return devices_config
    
    def _setup_hardware(self):
        monitor = monitors.Monitor("monitor", distance=MONITOR_DISTANCE, width=MONITOR_WIDTH)
        monitor.setSizePix(MONITOR_RESOLUTION)
        self.win = visual.Window(
            size=(MONITOR_RESOLUTION),
            units="pix",
            screen = SCREEN,
            fullscr=FULLSCREEN,
            colorSpace="rgb",
            monitor="FOVEAL",
            color=[0, 0, 0],
        )
        self.win.setMouseVisible(False)
        
        self.io = iohub.launchHubServer(
            window=self.win,
            experiment_code="FovealDecoding",
            session_code=self.session_id,
            **self.devices_config,
        )
        
        self.keyboard = self.io.getDevice("keyboard")
        self.tracker = self.io.getDevice("tracker")
        
    def _setup_stimuli(self): 
        self.text_stim = visual.TextStim(
            self.win,
            text="Eye Tracker Calibration",
            pos=[0, 0],
            height=45,
            color="black",
            units="pix",
            colorSpace="named",
            wrapWidth=self.win.size[0] * 0.9,
        )
        
    def _setup_bonus_validation_procedure(self):    
        self.target_stim = TargetStim(
            self.win,
            radius=26,
            fillcolor=[0.5, 0.5, 0.5],
            edgecolor=[-1, -1, -1],
            edgewidth=1,
            dotcolor=[1, -1, -1],
            dotradius=8,
            units="pix",
            colorspace="rgb",
        )
        
        self.validation_proc = iohub.ValidationProcedure(
            self.win,
            target=self.target_stim,  # target stim
            positions="FIVE_POINTS",  # string constant or list of points
            randomize_positions=True,  # boolean
            expand_scale=1.5,  # float
            target_duration=1.5,  # float
            target_delay=1.0,  # float
            enable_position_animation=True,
            color_space="rgb",
            unit_type="pix",
            progress_on_key="",  # str or None
            gaze_cursor=(-1.0, 1.0, -1.0),  # None or color value
            show_results_screen=True,  # bool
            save_results_screen=False,  # bool, only used if show_results_screen == True
        )
    
    def run_tracker_setup(self):
        self._setup_hardware()
        self._setup_stimuli()
        
        self.text_stim.draw()
        self.win.flip()

        hideWindow(self.win)
        result = self.tracker.runSetupProcedure()
        print("Calibration returned: ", result)
        showWindow(self.win)
        
        self.run_gaze_on_screen()
        return self.tracker, self.io

    def run_bonus_validation(self):    
        self._setup_bonus_validation_procedure()
        self.validation_proc.run()
        if self.validation_proc.results:
            results = self.validation_proc.results
            print("++++ Validation Results ++++")
            print("Passed:", results["passed"])
            print("failed_pos_count:", results["positions_failed_processing"])
            print("Units:", results["reporting_unit_type"])
            print("min_error:", results["min_error"])
            print("max_error:", results["max_error"])
            print("mean_error:", results["mean_error"])
        else:
            print("Validation Aborted by User.")
        
    def run_gaze_on_screen(self):    
        TRIAL_COUNT = 1
        T_MAX = 60.0

        gaze_ok_region = visual.Circle(
            self.win, lineColor="black", radius=170, units="pix", colorSpace="named"
        )

        gaze_dot = visual.GratingStim(
            self.win,
            tex=None,
            mask="gauss",
            pos=(0, 0),
            size=(25, 25),
            color="green",
            colorSpace="named",
            units="pix",
        )

        text_stim_str = "Eye Position: %.2f, %.2f. In Region: %s\n"
        text_stim_str += "Press space key to start next trial."
        missing_gpos_str = "Eye Position: MISSING. In Region: No\n"
        missing_gpos_str += "Press space key to start next trial."
        self.text_stim.setText(text_stim_str)

        t = 0
        while t < TRIAL_COUNT:
            self.io.clearEvents()
            self.tracker.setRecordingState(True)
            run_trial = True
            tstart_time = core.getTime()
            while run_trial is True:
                gpos = self.tracker.getLastGazePosition()
                valid_gaze_pos = isinstance(gpos, (tuple, list))
                gaze_in_region = valid_gaze_pos and gaze_ok_region.contains(gpos)
                if valid_gaze_pos:
                    if gaze_in_region:
                        gaze_in_region = "Yes"
                    else:
                        gaze_in_region = "No"
                    self.text_stim.text = text_stim_str % (gpos[0], gpos[1], gaze_in_region)

                    gaze_dot.setPos(gpos)
                else:
                    self.text_stim.text = missing_gpos_str

                gaze_ok_region.draw()
                self.text_stim.draw()
                if valid_gaze_pos:
                    gaze_dot.draw()

                flip_time = self.win.flip()

                if self.keyboard.getPresses(keys=" "):
                    run_trial = False
                elif core.getTime() - tstart_time > T_MAX:
                    run_trial = False
            self.tracker.setRecordingState(False)
            t += 1

        return

    def initiate_recording(self):
        self.io.clearEvents()
        self.tracker.setRecordingState(True)

    def stop_recording(self):
        self.tracker.setRecordingState(False)
