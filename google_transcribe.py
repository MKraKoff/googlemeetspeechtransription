import pyaudio
import queue
from google.cloud import speech
import pyautogui
import os

# Set the path to your credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google.json"


RATE = 16000  # Audio sample rate in Hz
CHUNK = 1000  # 100ms chunks for audio streaming

class AudioStream:
    """Handles live audio streaming."""

    def __init__(self, rate, chunk, input_device_index=None):
        self.rate = rate
        self.chunk = chunk
        self.input_device_index = input_device_index
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self._buff = queue.Queue()

    def list_devices(self):
        """Lists available input devices."""
        print("Available input devices:")
        for i in range(self.audio_interface.get_device_count()):
            info = self.audio_interface.get_device_info_by_index(i)
            print(f"{i}: {info['name']} - {info['maxInputChannels']} channels")

    def start_stream(self):
        """Starts the audio stream."""
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,  # Mono audio
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.input_device_index,
            stream_callback=self._fill_buffer,
        )

    def stop_stream(self):
        """Stops the audio stream."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Callback to fill the buffer with audio data."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Generator yielding chunks of audio data."""
        while True:
            yield self._buff.get()

def send_text_to_chatgpt(sentence):
    """
    Sends the transcribed text to the ChatGPT input box using pyautogui.
    """
    try:
        print(f"Sending to ChatGPT: {sentence}")
        pyautogui.typewrite(sentence)  # Type the text
        pyautogui.hotkey("shift", "enter")  # Press Shift + Enter for a line break
    except Exception as e:
        print(f"Error in sending text: {e}")

def transcribe_with_google_live(audio_stream):
    """
    Transcribes audio in real-time using Google Cloud Speech-to-Text API.
    """
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        model="default",
        enable_automatic_punctuation=True,  # Enable punctuation for better readability
        use_enhanced=False,  # This indicates using the default model, logged if enabled
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True  # Enable interim results if needed
    )

    def request_generator():
        """Yields audio data to the Google Cloud Speech API."""
        for audio_chunk in audio_stream.generator():
            yield speech.StreamingRecognizeRequest(audio_content=audio_chunk)

    responses = client.streaming_recognize(config=streaming_config, requests=request_generator())

    try:
        for response in responses:
            if not response.results:
                continue

            # Process each result
            result = response.results[0]
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                print(f"Transcription: {transcript}")  # Print transcription to console
                send_text_to_chatgpt(transcript)  # Send transcription to ChatGPT
    except Exception as e:
        print(f"Error during transcription: {e}")

def main():
    """
    Main function to capture audio and transcribe using Google Cloud Speech-to-Text in live mode.
    """
    stream = AudioStream(rate=RATE, chunk=CHUNK)
    stream.list_devices()

    # Select the input device
    device_index = int(input("Select the input device index for system audio: "))
    stream.input_device_index = device_index

    # Start the audio stream
    stream.start_stream()
    print("Listening... (Press Ctrl+C to stop)")

    try:
        transcribe_with_google_live(stream)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stream.stop_stream()

# Start transcription
if __name__ == "__main__":
    main()
