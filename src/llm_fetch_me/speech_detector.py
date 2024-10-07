#!/usr/bin/env python3

import rospy
import numpy as np
import sounddevice as sd
import threading
import queue
from std_msgs.msg import String  # Import ROS message type for string publishing
from faster_whisper import WhisperModel
import torch

# Global variables
audio_queue = queue.Queue()
is_recording = True
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "tiny.en"

def audio_callback(indata, frames, time, status):
    if status:
        rospy.logwarn(status)
    audio_queue.put(indata.copy())

def record_audio():
    """ Captures audio from the microphone and places it into a queue. """
    global is_recording
    chunk_duration = 0.2  # Chunk duration of 0.2 seconds for frequent updates
    sample_rate = 16000
    
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=int(sample_rate * chunk_duration), device=9):
        rospy.loginfo("Recording started...")
        while not rospy.is_shutdown() and is_recording:
            sd.sleep(100)  # Sleep in milliseconds

def transcribe_audio():
    """ Transcribes audio data from the queue and publishes the recognized text. """
    global is_recording, model
    buffer = np.array([], dtype=np.float32)
    
    # Create a ROS publisher
    pub = rospy.Publisher('speech_recognition/transcription', String, queue_size=10)
    
    while not rospy.is_shutdown() and is_recording:
        try:
            chunk = audio_queue.get(timeout=0.1)
            buffer = np.concatenate((buffer, chunk.flatten()))
            
            # Process when buffer reaches about 1 second of audio
            if len(buffer) >= 16000:
                if np.max(np.abs(buffer)) > 0:
                    buffer = buffer / np.max(np.abs(buffer))  # Normalize the audio buffer
                
                # Transcribe the buffer
                segments, info = model.transcribe(buffer, beam_size=1, language='en')
                for segment in segments:
                    text = segment.text.strip()  # Transcribed text
                    rospy.loginfo(f"Transcribed Text: {text}")
                    pub.publish(text)  # Publish the transcribed text to the ROS topic
                
                # Keep a small overlap for context
                buffer = buffer[-4000:]  # Retain the last 0.25 seconds of audio for smoother context transitions
        except queue.Empty:
            continue

def main():
    global is_recording, model
    # Initialize the ROS node
    rospy.init_node('speech_recognition_node', anonymous=True)
    
    rospy.loginfo("Loading Whisper model...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    rospy.loginfo(f"Model loaded using device: {device}")
    
    # Start recording and transcription in separate threads
    record_thread = threading.Thread(target=record_audio)
    transcribe_thread = threading.Thread(target=transcribe_audio)
    
    record_thread.start()
    transcribe_thread.start()
    
    # Keep running until ROS is shut down
    rospy.spin()

    # Stop recording when the node is shutting down
    is_recording = False
    record_thread.join()
    transcribe_thread.join()
    rospy.loginfo("Shutting down speech recognition node.")

if __name__ == "__main__":
    main()

