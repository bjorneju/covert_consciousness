import random 
import pygame
import serial
import datetime
import os
import tkinter as tk
from tkinter import ttk  # Import ttk from tkinter
from tkinter import messagebox  # Import messagebox
from collections import Counter
import time

bad_sounds = [
    'F2_40', 'F2_71',
    'F3_13', 'F3_18', 'F3_63', 'F3_71', 'F3_80', 'F3_81', #71
    'F4_16', 'F4_22', 'F4_27', 'F4_31', 'F4_36', 'F4_37', 'F4_45', 'F4_78', 'F4_79', 'F4_82', 'F4_87', 'F4_90',
    'M1_8',
    'M2_26', 'M2_28', 'M2_31', 
    'M3_23', 'M3_35', 'M3_71', 'M3_91',
    'M4_86'
]
bad_sounds = [s+'.wav' for s in bad_sounds]

### Functions to run the main experiment
# Main wrapper to run the experiment
def run_experiment(stimuli=None):

    # initialize serial com port
    global serialport 
    serialport = serial.Serial('COM3', 9600)

    # initialize pygame
    pygame.init()

    # initalize the audio player
    pygame.mixer.init()

    # input user ID
    user_ID, continuation, stimulus_seed, n_sessions, first_block = get_user_input("initialization")

    if continuation:
        print('deal with this!')
    else:
        if stimuli is None:
            # create stimuli
            stimuli = create_experiment_stimuli(n_sessions, seed=stimulus_seed)

        # run sessions
        for i, stimulus_set in enumerate(stimuli):
            print(i)
            print(first_block)
            print(i<first_block)
            if not i<first_block:
                start = get_user_input("session")
                stim_times = run_session(stimulus_set['stimuli'], stimulus_set['triggers'])


# Run a single session
def run_session(stimuli, triggers):
    
    stim_times = []  # Start time before trigger is sent
    try:
        for trigger, sound in zip(triggers, stimuli):  # Iterate over all sounds in a flattened list

            # Check if the current sound is a 'BREAK' marker
            break_time = check_for_break(sound)
            
            # Load sound stimulus
            pygame.mixer.music.load(sound)

            # Send triggers right before the sound plays
            send_trigger(trigger)

            # play the stimulus
            start_time = datetime.datetime.now()
            pygame.mixer.music.play()
            
            responses = []  # List to store all click responses
            max_response_end_time = start_time + datetime.timedelta(seconds=8)
            
            while pygame.mixer.music.get_busy() and datetime.datetime.now() <= max_response_end_time:
                1
                '''
                # look for responses while stimulus is playing
                if event.button == 1:  # Left click for 'yes'
                    send_trigger(1)  # Send trigger for 'yes' click
                else:  # Any other button for 'no'
                    send_trigger(2)  # Send trigger for 'no' click
                '''
            # Wait 50 ms after the sound ends to start next sound
            end_time = datetime.datetime.now()  # Start time before trigger is sent
            pygame.time.wait(50)
            stim_times.append(end_time-start_time)
                    
    except Exception as e:
        print(str(e))
    finally:
        print("Session finished")

    return stim_times

### TRIGGER FUNTIONS 
# Function to create triggers for all stimulus types
def set_triggers(stimuli):
    # trigger mappings
    first_trigger_mapping = {
        'silent': 1,
        'beep_breaks': 2,
        'action_command': 3,
        'meaningful': 4,
        'scrambled': 5,
        'action_beeps': 6,
    }
    
    beep_break_mapping = {
        'beep_A': 11,
        'beep_C': 12,
        'beep_E': 13,
        'last-up': 14,
        'no-local': 15,
        'beeps': 16,
    }
    
    beep_break_type = {
        'standard': 21,
        'deviant': 22,
    }
    
    voice_mapping = {
        'M1': 31,
        'M2': 32,
        'M3': 33,
        'M4': 34,
        'M5': 35,
        'M6': 36,
        'F1': 37,
        'F2': 38,
        'F3': 39,
        'F4': 40,
        'F5': 41,
        'F6': 42,
    }
    
    action_limb_mapping = {
        'hand': 51,
        'foot': 52,
        'rest': 53,
    }
    
    action_side_mapping = {
        'left': 61,
        'right': 62,
    }

    triggers = []
    for s in stimuli:
        # set first trigger (trial type)
        condition = s.split('/')[-2]
        first_trigger = first_trigger_mapping[condition]
        
        # set second trigger (trial type detail 1)
        if condition=='beep_breaks':
            stimulus = s.split('/')[-1].split('.')[0]
            if 'p3b' in stimulus:
                second_trigger = beep_break_mapping[stimulus.split('_')[1]]
            else:
                second_trigger = beep_break_mapping[stimulus]
        elif condition in ['meaningful', 'scrambled']:
            stimulus = s.split('/')[-1].split('_')[0]
            second_trigger = voice_mapping[stimulus]
        elif condition == 'action_command':
            stimulus = s.split('/')[-1].split('_')[-1].split('.')[0]
            second_trigger = action_limb_mapping[stimulus]
        elif condition == 'action_beeps':
            stimulus = s.split('/')[-1].split('_')[-1].split('.')[0]
            second_trigger = beep_break_mapping[stimulus]
        else:
            second_trigger = None

        # set third trigger (trial type detail 2)
        if condition == 'action_command':
            try:
                stimulus = s.split('/')[-1].split('_')[2]
                stimulus = s.split('/')[-1].split('_')[1]
                third_trigger = action_side_mapping[stimulus]
            except:
                None
        else:
            third_trigger = None

        triggers.append((first_trigger, second_trigger, third_trigger))
        
    return triggers


# Function to send triggers based on the stimulus delivered
def send_trigger(trigger):
    # send first trigger
    serialport.write(trigger[0].to_bytes(length=1, byteorder="big"))
    time.sleep(.10)

    #send second trigger
    if not trigger[1] is None:
        serialport.write(trigger[1].to_bytes(length=1, byteorder="big"))
        time.sleep(.10)
    
    #send third trigger
    if not trigger[2] is None:
        serialport.write(trigger[2].to_bytes(length=1, byteorder="big"))
        time.sleep(.10)
        
### STIMULUS DEFINITION FUNCTIONS    
# Function to set paths for all files (creates global variables used elsewhere)
def get_stimulus_paths(processed_path, seed=42):
    
    global beeps, meaningful_stimuli, scrambled_stimuli, action_stimuli, action_beeps, silent, p3b_lastup, p3b_noup
    
    # silent
    silent = [processed_path+'/silent/silence.wav']
    
    # beep breaks
    beeps = [processed_path+'/beep_breaks/beep_C.wav']
    
    # p3b
    p3b_lastup = [processed_path+'/beep_breaks/p3b_last-up.wav']
    p3b_noup = [processed_path+'/beep_breaks/p3b_no-local.wav']
    
    # semantic
    random.seed(seed)
    path = processed_path+'/meaningful'
    meaningful_stimuli = [path + '/' + f for f in os.listdir(path) if not f.startswith('.')]
    
    path = processed_path+'/scrambled'
    scrambled_stimuli = [path + '/' + f for f in os.listdir(path) if not f.startswith('.')]
    random.shuffle(scrambled_stimuli)
    
    # action commands
    path = processed_path+'/action_command'
    action_stimuli = [path + '/' + f for f in os.listdir(path) if not f.startswith('.')]

    # action beeps
    path = processed_path+'/action_beeps'
    action_beeps = [path + '/' + f for f in os.listdir(path) if not f.startswith('.')]



# Function to create the stimuli (file names) for a single session 
def create_session_stimuli(beeps, used_meaningful, used_scrambled, seed=42):
    
    # CREATE STIMULI FOR ONE CONDITION
    n_silent = 5
    n_runin = 5
    n_meaningful = 20
    n_scrambled = 20
    n_right = 20
    n_left = 0
    n_foot = 0
    n_rest = 20
    
    random.seed(seed)

    # pick meaningful stimuli among unused meaningful stimuli 
    unused_meaningful = [m for m in meaningful_stimuli if m not in used_meaningful and m not in used_scrambled and m not in bad_sounds]
    meaningful = random.choices(unused_meaningful, k=n_meaningful)
    used_meaningful.extend(meaningful)

    # pick scrambled stimuli that correspond to the picked meaningful stimuli, but with opposite gender
    unused_scrambled = [s for s in scrambled_stimuli if s not in used_scrambled]
    valid_scrambled = [s for s in unused_scrambled if not sum([s[1:] == m[1:] for m in meaningful]) and not s in used_meaningful]
    scrambled = random.choices(valid_scrambled, k=n_scrambled)
    used_scrambled.extend(scrambled)

    # pick specific action stimuli
    # right hand
    right_stimuli = random.choices([s for s in action_stimuli if 'right' in s], k=n_right)
    # left hand
    left_stimuli = random.choices([s for s in action_stimuli if 'left' in s], k=n_left)
    # foot
    foot_stimuli = random.choices([s for s in action_stimuli if 'foot' in s], k=n_foot)
    # rest
    rest_stimuli = random.choices([s for s in action_stimuli if 'rest' in s], k=n_rest)
    
    trial_stimuli = (
        n_silent*silent + 
        meaningful[n_runin:] + 
        scrambled + 
        right_stimuli +
        left_stimuli +
        foot_stimuli +
        rest_stimuli
    )
    random.shuffle(trial_stimuli)
    trial_stimuli = meaningful[:n_runin] + trial_stimuli
    
    # beep break stimuli
    beeps = len(trial_stimuli)*beeps
    
    # merge stimuli
    merged_stimuli = [s for stim in zip(trial_stimuli, beeps) for s in stim]
    
    # add action beeps after each action block
    all_stimuli = []
    for stimulus in merged_stimuli:
        all_stimuli.append(stimulus)
        if 'action_command' in stimulus:
            all_stimuli.append(action_beeps[0])
           
    return all_stimuli, used_meaningful, used_scrambled

# Function to create the stimuli (file names) for a single session 
def create_pilot_session_stimuli(i, standard, deviant, seed=42, n_action = 20, n_standard = 20, n_deviant = 5, n_runin = 3):
    
    # CREATE STIMULI FOR ONE CONDITION
    n_total = n_standard + n_action * (n_standard + n_deviant + 1)
    random.seed(seed)
    actions_right = [a for a in action_stimuli if 'right' in a]
    actions_rest = [a for a in action_stimuli if 'rest' in a]
    p3b_blocks = []
    for i in range(n_action):
        if i%2==0:
            p3b = n_deviant*deviant+n_standard*standard
            random.shuffle(p3b)
            act = random.choice(actions_right)
            p3b_blocks.extend(n_runin*standard+p3b+[act])
        else:
            p3b = n_deviant*deviant+n_standard*standard
            random.shuffle(p3b)
            act = random.choice(actions_rest)
            p3b_blocks.extend(n_runin*standard+p3b+[act])
        
    trial_stimuli = (
        n_standard*standard +
        [random.choice(actions_rest)] +
        p3b_blocks
    )
    
    # add action beeps after each action block
    all_stimuli = []
    for stimulus in trial_stimuli:
        all_stimuli.append(stimulus)
        if 'action_command' in stimulus:
            all_stimuli.append(action_beeps[0])
           
    return all_stimuli

# Function to create stimuli lists for multiple sessions (for a full experiment)
def create_experiment_stimuli(n_sessions, processed_path, seed=42):

    # ensure paths are available
    get_stimulus_paths(processed_path, seed)
    
    # create stimuli and triggers for all sessions
    experiment_stimuli = []
    used_scrambled = []
    used_meaningful = []
    for i in range(n_sessions):
        stimuli, used_meaningful, used_scrambled = create_session_stimuli(beeps, used_meaningful, used_scrambled, seed)
        triggers = set_triggers(stimuli)
        experiment_stimuli.append({
            'stimuli': stimuli,
            'triggers': triggers
        })

    return experiment_stimuli

def create_pilot_experiment_stimuli(n_sessions, processed_path, seed=42, n_action = 15, n_standard = 16, n_deviant = 4):

    # ensure paths are available
    get_stimulus_paths(processed_path, seed)
    
    # create stimuli and triggers for all sessions
    p3b = []
    experiment_stimuli = []
    for i in range(n_sessions):
        standard = p3b_lastup if i%2==1 else p3b_noup
        deviant = p3b_lastup if i%2==0 else p3b_noup
        stimuli = create_pilot_session_stimuli(i, standard, deviant, n_action = n_action, n_standard = n_standard, n_deviant = n_deviant)
        triggers = set_triggers(stimuli)
        experiment_stimuli.append({
            'stimuli': stimuli,
            'triggers': triggers
        })

    return experiment_stimuli

### Functions needed to run the experiment (and front end)
# Tkinter dialog box functions
def get_user_input(case):

    font="Helvetica"
    font_size=12
    font_vars=(font, font_size)

    #to get subject ID and check if it's a new run or a continuation
    if case=='initialization':
        root = tk.Tk()
        root.title("Experiment Setup")
    
        # Styling
        root.geometry("400x400")
        style = ttk.Style()
        style.theme_use('clam')  # You can experiment with different themes like 'alt', 'clam', 'classic', etc.
    
        tk.Label(root, text="Welcome! Please type the subject ID:", font=font_vars).pack(pady=10)
    
        # Subject ID entry using ttk
        subject_ID_entry = ttk.Entry(root, font=font_vars, width=30)
        subject_ID_entry.pack(pady=5)
        subject_ID_entry.focus_set()
    
        # Continuation checkbox using ttk
        is_continuation = False
        
        tk.Label(root, text="Which block should we start from?", font=font_vars).pack(pady=10)
        start_trial = ttk.Entry(root, font=font_vars, width=10)
        start_trial.insert(0, "1") 
        start_trial.pack(pady=10)
    
        # "New stimulus set" checkbox using ttk
        tk.Label(root, text="Seed for stimulus generation:", font=font_vars).pack(pady=10)
        new_stimuli = ttk.Entry(root, font=font_vars, width=10)
        new_stimuli.insert(0, "42") 
        new_stimuli.pack(pady=10)
        
        # Set number of trials to create
        tk.Label(root, text="Number of sessions:", font=font_vars).pack(pady=10)
        n_sessions = ttk.Entry(root, font=font_vars, width=10)
        n_sessions.insert(0, "8") 
        n_sessions.pack(pady=10)
        # Function to handle submit
        def on_submit():
            global subject_ID, continuation_state, stimulus_seed, num_sessions, first_block
            subject_ID = subject_ID_entry.get()
            #continuation_state = is_continuation.get()
            stimulus_seed = new_stimuli.get()
            num_sessions = n_sessions.get()
            first_block = start_trial.get()
            root.destroy()
    
        submit_button = ttk.Button(root, text="Submit", command=on_submit)
        submit_button.pack(pady=10)
    
        # Run the main loop and wait for input
        root.mainloop()
        
        # Return the values after the window is closed
        return subject_ID, is_continuation, stimulus_seed, int(num_sessions), int(first_block)

    # Dialog box for starting new session
    elif case=='session':
        root = tk.Tk()
        root.title("Session beginning")
    
        # Styling
        root.geometry("400x400")
        style = ttk.Style()
        style.theme_use('clam')  # You can experiment with different themes like 'alt', 'clam', 'classic', etc.
    
        tk.Label(root, text="Press start when ready for the next session", font=font_vars).pack(pady=10)
    
        def on_submit():
            root.destroy()
    
        submit_button = ttk.Button(root, text="Start", command=on_submit)
        submit_button.pack(pady=10)
    
        # Run the main loop and wait for input
        root.mainloop()
        
        # Return the values after the window is closed
        return True

    # Dialog box for pausing the script (maybe not used)
    elif case=='pause':
        root = tk.Tk()
        root.title("Pause session")
    
        # Styling
        root.geometry("400x400")
        style = ttk.Style()
        style.theme_use('clam')  # You can experiment with different themes like 'alt', 'clam', 'classic', etc.
    
        tk.Label(root, text="Press to pause the session (after next sound)", font=font_vars).pack(pady=10)
    
        def on_submit():
            root.destroy()
    
        submit_button = ttk.Button(root, text="Pause", command=on_submit)
        submit_button.pack(pady=10)
    
        # Run the main loop and wait for input
        root.mainloop()
        
        # Return the values after the window is closed
        return True

### Function to check for break (not used yet?)
def check_for_break(sound):
    
    if sound == "BREAK":
        # Add a break entry to the results
        return datetime.datetime.now()
    return None
