import librosa
import numpy as np
import pywt
import soundfile as sf
import random
import os
from tqdm import tqdm
#import noisereduce as nr


### Organizing audio stimuli
def organize_semantic_stimuli(raw_path, processed_path, segment_ms=300, fade_ms=150):
    for folder in os.listdir(raw_path):
        if not (folder == 'action_commands' or folder[0]=='.'): 
            for file in tqdm(os.listdir(raw_path + '/' + folder), desc='Processing raw files from {}'.format(folder)):
                # get shuffled sound
                if not file[0]=='.':
                    sentence, scrambled, rate = smooth_shuffle_audio_segments(raw_path + '/' + folder + '/' + file, segment_ms, fade_ms)
                    split = file.split(' ')
                    file_name = split[0] + '_' + split[1]
                    sf.write(processed_path + '/meaningful/' + file_name, sentence, rate)
                    sf.write(processed_path + '/scrambled/' + file_name, scrambled, rate)

def organize_action_stimuli(raw_path, processed_path):
    # first pick out and create task command sounds
    tasks = dict()
    for file in os.listdir(raw_path+'/action_commands'):
        name = file.split(' ')[0]
        if '1.wav' in file:
            stimulus, sr = librosa.load(raw_path + '/action_commands' + '/' + file, sr=None)
            tasks[name] = stimulus
             
    # next the actual command sounds
    commands = dict()
    aux = dict()
    for file in os.listdir(raw_path+'/action_commands'):
        name = file.split(' ')[0]
        
        if name not in commands:
            commands[name] = {}
        if not '1.wav' in file:
            stimulus, rate = librosa.load(raw_path + '/action_commands' + '/' + file, sr=None)
            commands[name][file] = stimulus
    name_mapping = {
        '2.wav': 'foot.wav',
        '3.wav': 'right_hand.wav',
        '4.wav': 'left_hand.wav',
        '5.wav': 'rest.wav',
    }
    
    # create silent periods
    # silence pre
    silence_pre = np.zeros(int(0.2*rate))
    silence_mid = np.zeros(int(0.5*rate))
    silence_baseline = np.zeros(int(0.8*rate))
    
    # combine waveforms
    for name, task in tasks.items():
        if name in commands:
            for file, command in commands[name].items():
                split = file.split(' ')
                file_name = split[0] + '_' + name_mapping[split[1]]
                sound = np.concatenate((silence_pre, task, silence_mid, command, silence_baseline))
                sf.write(processed_path + '/action_command/' + file_name, sound, rate)
            

    # create beep signal
    frequencies = [220, 220*2+1, 220*3-1, 220*6+2]  # Fundamental and overtones for A (below middle C)
    
    # Generate waveforms
    beep_duration=0.20
    t = np.linspace(0, beep_duration, int(rate * beep_duration), endpoint=False)
    beep = sum(np.sin(2 * np.pi * f * t) for f in frequencies)
    silence_active = np.zeros(int(rate*4))

    # combine waveforms
    action_sound = np.concatenate((beep,silence_active,beep))
    
    # save files
    sf.write(processed_path+'/action_beeps/beeps.wav', normalize_sound(action_sound), rate)
    

def organize_beep_block_stimuli(processed_path, rate=96000):
    # Fundamental frequencies for Middle C, G, and E, with some overtones
    A_frequencies = [220, 220*2, 220*3, 220*6]  # Fundamental and overtones for A (below middle C)
    C_frequencies = [261.63, 261.63*2, 261.63*3, 261.63*6]  # Fundamental and overtones for middle C
    E_frequencies = [329.63, 329.63*2, 329.63*3, 329.63*6]  # Fundamental and overtones for E

    # single beep stimulus
    # Generate waveforms
    beep_C = generate_beep(frequencies=C_frequencies,sampling_rate=rate)
    
    # save files
    sf.write(processed_path+"/beep_breaks/beep_C.wav", normalize_sound(beep_C), rate)


def organize_p3b_block_stimuli(processed_path, rate=96000):
    # Fundamental frequencies for Middle C, G, and E, with some overtones
    A_frequencies = [220, 220*2, 220*3, 220*6]  # Fundamental and overtones for A (below middle C)
    C_frequencies = [261.63, 261.63*2, 261.63*3, 261.63*6]  # Fundamental and overtones for middle C
    E_frequencies = [329.63, 329.63*2, 329.63*3, 329.63*6]  # Fundamental and overtones for E

    # single beep stimulus
    # Generate waveforms
    beeps_lastup = generate_beeps(sampling_rate=rate, silence_duration=0.09, beep_duration=0.09, frequencies=C_frequencies, last_frequencies=E_frequencies, end_silence=0.5)
    beeps_nolocal = generate_beeps(sampling_rate=rate, silence_duration=0.09, beep_duration=0.09, frequencies=C_frequencies, last_frequencies=None, end_silence=0.5)
    
    # save files
    sf.write(processed_path+"/beep_breaks/p3b_last-up.wav", normalize_sound(beeps_lastup), rate)
    sf.write(processed_path+"/beep_breaks/p3b_no-local.wav", normalize_sound(beeps_nolocal), rate)


def organize_silent_stimuli(processed_path, rate=96000, duration=4):
    
    # Generate waveform
    silence = np.zeros(duration*rate)
    
    # save file
    sf.write(processed_path+"/silent/silence.wav", silence, rate)

### Creating shuffled voice stimuli

# Main function for shuffling a audio file 
def smooth_shuffle_audio_segments(filename, segment_ms, fade_ms):
    """
    Shuffle segments of an audio signal randomly with smoothing (cross-fading) applied
    between segments.
    
    Parameters:
        audio (numpy.ndarray): The input audio signal.
        segment_ms (int): The length of each segment in milliseconds.
        fade_ms (int): The length of the cross-fade in milliseconds.
        
    Returns:
        truncated_audio (numpy.ndarray): The original audio signal, with silent periods on the start/end trimmed away.
        shuffled_audio (numpy.ndarray): The shuffled and smoothed audio signal.
        sr (int): The sampling rate of the signal.
    """

    # Load, crop and pad audio file
    audio, sr = librosa.load(filename, sr=None)
    
    # Calculate the number of samples per segment and fade
    segment_length = int((segment_ms / 1000.0) * sr)
    fade_length = int((fade_ms / 1000.0) * sr)
    
    # Ensure the fade length is not longer than the segment length
    fade_length = int(min(fade_length, segment_length // 2))
    fade_samples = int(fade_length / 2)
    
    # Calculate the number of complete segments (counting "overlap")
    num_segments = len(audio) // (segment_length + fade_samples)
    
    # Truncate the audio to a multiple of the segment length
    audio_c = audio.copy()
    truncated_audio = audio_c[:num_segments * (segment_length + fade_samples) - fade_samples]

    if num_segments<3:
        print('very short audio, not shuffling!')
        return truncated_audio, truncated_audio, sr
    
    # Reshape the audio into segments
    audio_segments = []
    a=0
    for i in range(num_segments):
        if i==0:
            first_segment = list(
                truncated_audio[:segment_length+fade_samples]
            )
        elif i==num_segments-1:
            last_segment =  list(
                truncated_audio[-(segment_length+fade_samples):]
            )
        else:
            start = i*segment_length - fade_samples
            stop = (i+1)*segment_length + fade_samples
            audio_segments.append(list(
                truncated_audio[start:stop]
            ))
    
    # Shuffle the segments between first and last
    audio_segments.reverse()
    audio_segments = [first_segment, *audio_segments, last_segment]

    # Stitch together randomized segments, using the "optimal" overlap algorithm 
    shuffled_audio = np.array(audio_segments[0][:segment_length])
    for i in range(1, len(audio_segments)):
        # pick out overlapping part
        previous_end = np.array(audio_segments[i-1][-fade_samples:])
        next_start = np.array(audio_segments[i][:fade_samples])

        # find the n (fade_samples) consequtive samples that are most similar 
        index = find_most_similar_window(previous_end, next_start)
        shuffled_audio = np.hstack((
            shuffled_audio, 
            audio_segments[i-1][-fade_samples:-(fade_samples-index)], 
            audio_segments[i][index:fade_samples+segment_length]
        ))
    
    return audio, shuffled_audio, sr # previously truncated_audio

# loading, cropping, and padding
def rough_preprocessing(filename, length=5, padding=0.1):
    # Load audio file
    audio, sr = librosa.load(filename, sr=None)

    # crop silent periods before and after voice to shorten file
    # audio = crop_audio(audio, sr)

    return audio, sr

# Function to crop the silent periods from start/end of an audio signal
def crop_audio(sound, sr, margin=0.1, threshold=0.01, surrounding_sec=0.05, surrounding_threshold=0.1):
    """
    Crop the audio signal to remove silence before and after the main sound,
    considering a 1-second surrounding window for the threshold condition.
    
    Parameters:
    - sound (numpy.array): The waveform of a sound.
    - sr (int): Sampling rate of the sound file.
    - margin (float=0.1): Maximum time [s] to include before the first, and
                          after the last, non-silent part.
    - threshold (float=0.05): Amplitude threshold for silence detection.
    - surrounding_sec (int=1): The duration of the surrounding window in seconds.
    - surrounding_threshold (float=0.3): Proportion of samples within the window
                                         that must exceed the threshold.
    
    Returns:
    - Cropped numpy array of the audio signal.
    """
    # Calculate the window size in samples
    window_size = int(surrounding_sec * sr)
    
    # Find where the signal is above the threshold
    above_threshold = np.abs(sound) > threshold * np.max(np.abs(sound))
    
    # Calculate the moving average of the above_threshold array
    moving_avg = np.convolve(above_threshold, np.ones(window_size)/window_size, mode='full')
    
    # Find indices where the moving average exceeds the surrounding threshold
    true_indices = np.where(moving_avg > surrounding_threshold)[0]
    
    if len(true_indices) == 0:
        # If the condition is never met, return the original signal
        return sound
    
    # Determine the start and end indices with margin
    margin = max(margin, surrounding_sec/2)
    start_index = max(true_indices[0] - int(margin * sr), 0)
    end_index = min(true_indices[-1] + int(margin * sr), len(sound))
    
    # Crop the audio signal
    cropped_audio = sound[start_index:end_index]
    
    return cropped_audio

# Function to downsample too long sentences
def remove_evenly_spaced_indices(original_audio, max_audio_length):
    N = len(original_audio)
    M = N - max_audio_length
    
    # Calculate approximate spacing between indices to remove
    spacing = N / M
    
    # Generate indices to remove - round to nearest integer and ensure unique values
    indices_to_remove = np.unique(np.round(np.arange(0, N, spacing)).astype(int))
    
    # Limit the number of indices to remove to M, in case rounding causes more indices
    indices_to_remove = indices_to_remove[:M]
    
    # Use np.delete() to remove the calculated indices
    downsampled_audio = np.delete(original_audio, indices_to_remove)
    
    return downsampled_audio

# force all sentences to be no more than "length" seconds long
def force_length_of_audio(audio, sr, length, min_padding=0.1):

    audio_length = len(audio)
    max_audio_length = sr*length-int(2*min_padding*sr)
    
    if audio_length>max_audio_length:
        audio = remove_evenly_spaced_indices(audio, max_audio_length)
        
    long_audio = np.zeros(int(len(audio)+2*min_padding*sr))
    padding = int(sr*min_padding)

    long_audio[padding:-(padding)] = audio
    
    return long_audio

    
# Function to find where two segments are maximal overlapping for n consequtive samples    
def find_most_similar_window(previous_end, next_start, n_samples=5):
    """Find the start index of the most similar 5-sample window."""
    min_distance = float('inf')
    best_index = -1
    
    # Ensure the time series are at least 5 samples long
    if len(previous_end) < n_samples or len(next_start) < n_samples:
        print("Both series need to be at least n_samples samples long.")
        return best_index
    
    # Loop through each possible 5-sample window
    for i in range(len(previous_end) - (n_samples-1)):
        distance = euclidean_distance(previous_end[i:i+n_samples], next_start[i:i+n_samples])
        if distance < min_distance:
            min_distance = distance
            best_index = i
    
    return best_index+int(n_samples/2)


def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two arrays."""
    return np.sqrt(np.sum((a - b) ** 2))


### Function for creating "beep break" blocks
def generate_beep(sampling_rate=16000, block_duration=1.00, beep_duration=0.09, frequencies=[261.63]):
    """
    Generate a waveform with a single beep, starting and ending with silence.
    
    Parameters:
    - sampling_rate: The sample rate of the audio.
    - silence_duration: Duration of the starting and ending silence in seconds.
    - beep_duration: Duration of each beep in seconds.
    - frequencies: List of frequencies for the beeps. Assumes the list contains the fundamental frequency
                   and any desired overtones.
    - last_frequencies: The frequencies for the last beep, if different from the others.
    
    Returns:
    - A NumPy array containing the waveform.
    """
    # Generate silence
    waveform = np.zeros(int(sampling_rate * block_duration))
    
    # Generate beep
    t = np.linspace(0, beep_duration, int(sampling_rate * beep_duration), endpoint=False)
    beep = sum(np.sin(2 * np.pi * f * t) for f in frequencies)

    # Calculate the starting index for the beep to be in the middle of the silence
    start_index = int((len(waveform) - len(beep)) / 2)
    
    # Insert the beep into the silence
    waveform[start_index:start_index+len(beep)] = beep
    
    return waveform


def generate_beeps(sampling_rate=16000, silence_duration=0.09, beep_duration=0.09, frequencies=[261.63], last_frequencies=None, end_silence=0.001):
    """
    Generate a waveform with 5 beeps, starting and ending with silence.
    
    Parameters:
    - sampling_rate: The sample rate of the audio.
    - silence_duration: Duration of the starting and ending silence in seconds.
    - beep_duration: Duration of each beep in seconds.
    - frequencies: List of frequencies for the beeps. Assumes the list contains the fundamental frequency
                   and any desired overtones.
    - last_frequencies: The frequencies for the last beep, if different from the others.
    
    Returns:
    - A NumPy array containing the waveform.
    """
    # Generate silence
    silence = np.zeros(int(sampling_rate * silence_duration))
    end_silence = np.zeros(int(sampling_rate * end_silence))
    
    # Generate beeps
    t = np.linspace(0, beep_duration, int(sampling_rate * beep_duration), endpoint=False)
    beeps = [sum(np.sin(2 * np.pi * f * t) for f in frequencies) for _ in range(4)]
    
    # If a different frequency is specified for the last beep, generate it
    if last_frequencies is not None:
        last_beep = sum(np.sin(2 * np.pi * f * t) for f in last_frequencies)
        beeps.append(last_beep)  # Append the last beep with different frequencies
    else:
        beeps.append(beeps[0])  # Append a beep similar to the first ones if no last_frequencies provided
    
    # Normalize beeps to prevent clipping
    beeps = [beep / np.max(np.abs(beep)) for beep in beeps]
    
    # Concatenate beeps with intermediate silence
    beep_sequence = np.concatenate([np.concatenate([beep, silence]) for beep in beeps[:-1]] + [beeps[-1]])
    
    # Generate waveform with starting and ending silence
    waveform = np.concatenate([silence, beep_sequence, silence, end_silence])
    
    return waveform


#Function to normailze all sounds to max 1
def normalize_sound(waveform):
    return waveform/np.max(waveform)
