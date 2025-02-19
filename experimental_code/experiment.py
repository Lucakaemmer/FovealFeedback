# Imports
from experimental_condition import ExperimentalCondition
from experimental_localiser import Localizer
import pandas as pd

from eye_tracker import EyeTracker
from const import N_RUNS
import pylink
from psychopy import core, gui
from datetime import datetime
import json
import os


# Settings
debugging = False
data_collection = True


# Initiating data collection
if data_collection:
    info = {"Participant ID": "", "Age": "", "Gender": ""}
    dlg = gui.DlgFromDict(dictionary=info, title="Participant Information")
    if dlg.OK is False:
        core.quit()  # user pressed cancel
    info["Start Time"] = datetime.now().strftime("%d_%H%M%S")

    folder_name = f"{info['Participant ID']}_{info['Start Time']}"
    os.makedirs(folder_name, exist_ok=True)
    with open(f"{folder_name}/{info['Participant ID']}__Info", "w") as file:
        json.dump(info, file, indent=4)
else:
    folder_name = "test"


# Eye Tracking
if not debugging:
    eye = EyeTracker(f"{folder_name}/{info['Participant ID']}")
    tracker, io = eye.run_tracker_setup()
else:
    tracker = 0
    io = 0
    


# Defining Conditions
C1 = ExperimentalCondition(position="periphery", eye_trigger=True, tracker=tracker, io=io)
C2 = ExperimentalCondition(position="fovea", eye_trigger=False, tracker=tracker, io=io)

run = 0
# Executing Conditions
for i in range(N_RUNS):
    run += 1
    print(f"Executing Condition 1 Run {run}...")
    eye.initiate_recording()
    data, gaze_distances = C1.execute_run(mouse_stimulation=debugging, n_run=run)
    data.to_csv(f"{folder_name}/{info['Participant ID']}_data_C1_R{run}.csv")
    gaze_distances.to_csv(f"{folder_name}/{info['Participant ID']}_gaze_C1_R{run}.csv")
    eye.stop_recording()

    run += 1
    print(f"Executing Condition 2 Run {run}...")
    eye.initiate_recording()
    data, gaze_distances = C2.execute_run(mouse_stimulation=debugging, n_run=run)
    data.to_csv(f"{folder_name}/{info['Participant ID']}_data_C2_R{run}.csv")
    gaze_distances.to_csv(f"{folder_name}/{info['Participant ID']}_gaze_C2_R{run}.csv")
    eye.stop_recording()


# Executing localizers
loc = Localizer(tracker=tracker, io=io)

print("Executing Retinotopic Localizer ...")
eye.initiate_recording()
retinal_data = loc.retinotopic_localizer()
eye.stop_recording()
retinal_data.to_csv(f"{folder_name}/{info['Participant ID']}_retinal_data.csv")

print("Executing Object Localizer...")
eye.initiate_recording()
object_data = loc.object_localizer()
eye.stop_recording()
object_data.to_csv(f"{folder_name}/{info['Participant ID']}_object_data.csv")

# Closing up Experiment
if not debugging:
    tracker.setConnectionState(False)
core.quit()
