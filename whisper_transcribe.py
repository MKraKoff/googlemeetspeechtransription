import whisper
import pyaudio
import numpy as np
import queue
import pyautogui
import re
from config import MODEL, BUFFER_DURATION, OVERLAP_DURATION

# Load the Whisper model
model = whisper.load_model(MODEL)

# Audio recording parameters
RATE = 16000  # Whisper expects 16kHz audio
CHUNK = 1024  # Number of audio frames per buffer

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

#############################################
#         Tokenize & Edit Distance          #
#############################################

def tokenize(line):
    """
    Tokenize the input line into words, stripping punctuation and converting to lowercase.
    Returns both original tokens and normalized tokens.
    """
    # Regex to match words (including hyphenated) and retain their original forms
    original_tokens = re.findall(r"[A-Za-z0-9'\-]+", line)
    # Normalize tokens: lowercase
    normalized_tokens = [token.lower() for token in original_tokens]
    return original_tokens, normalized_tokens

def levenshtein(s1, s2):
    """
    Compute the Levenshtein (edit) distance between two strings.
    """
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]
    for i in range(len(s1)+1):
        dp[i][0] = i
    for j in range(len(s2)+1):
        dp[0][j] = j
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[len(s1)][len(s2)]

def approximate_match(t1_norm, t2_norm, max_distance=2):
    """
    Determine if two normalized tokens are approximately the same based on Levenshtein distance.
    """
    if t1_norm == t2_norm:
        return True
    return levenshtein(t1_norm, t2_norm) <= max_distance

#############################################
#     Overlap Detection Function            #
#############################################

def find_overlap_length(previous, current, max_distance=2, min_overlap=2):
    """
    Find the number of overlapping tokens between the end of 'previous' and the start of 'current'.
    Returns the length of the overlap.
    """
    _, previous_norm = tokenize(previous)
    _, current_norm = tokenize(current)

    max_possible = min(len(previous_norm), len(current_norm))
    max_possible = min(max_possible, 10)  # Limit to last 10 words for efficiency

    for overlap in range(max_possible, min_overlap-1, -1):
        suffix_prev = previous_norm[-overlap:]
        prefix_curr = current_norm[:overlap]
        match = True
        for tokA, tokB in zip(suffix_prev, prefix_curr):
            if not approximate_match(tokA, tokB, max_distance):
                match = False
                break
        if match:
            return overlap
    return 0  # No sufficient overlap found

#############################################
#     Merge and Send Transcription          #
#############################################

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

def transcribe_live():
    """
    Performs live transcription using Whisper and prints/sends the transcribed text.
    Implements merging to handle overlapping transcriptions.
    """
    stream = AudioStream(rate=RATE, chunk=CHUNK)
    stream.list_devices()

    # Select the input device
    try:
        device_index = int(input("Select the input device index for system audio: "))
    except ValueError:
        print("Invalid input. Using default input device.")
        device_index = None
    stream.input_device_index = device_index

    # Start the audio stream
    stream.start_stream()
    print("Listening... (Press Ctrl+C to stop)")

    previous_transcription = ""

    try:
        audio_buffer = b""
        chunk_size = int(RATE * BUFFER_DURATION)
        overlap_size = int(RATE * OVERLAP_DURATION)

        audio_gen = stream.generator()

        while True:
            # Collect audio chunks
            try:
                audio_chunk = next(audio_gen)
            except StopIteration:
                break  # Stream has ended

            audio_buffer += audio_chunk

            # Process when the buffer reaches the required size
            if len(audio_buffer) >= chunk_size * 2:  # Ensure enough audio for processing
                # Convert buffer to NumPy array
                audio_np = np.frombuffer(audio_buffer[:chunk_size], dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe using Whisper
                result = model.transcribe(audio=audio_np, fp16=False, temperature=0.0, language="en")
                transcription = result['text'].strip()

                print(f"Transcription: {transcription}")

                if transcription:
                    # Tokenize the transcription
                    original_tokens, normalized_current = tokenize(transcription)

                    # Merge with previous transcription to handle overlaps
                    overlap_len = find_overlap_length(previous_transcription, transcription, max_distance=2, min_overlap=2)

                    if overlap_len > 0:
                        # Remove the overlapping tokens from the beginning
                        new_part_original = ' '.join(original_tokens[overlap_len:])
                        if new_part_original:
                            send_text_to_chatgpt(new_part_original)
                        else:
                            print("No new content to send after removing overlap.")
                    else:
                        # No overlap; send the entire transcription
                        send_text_to_chatgpt(transcription)

                    # Update previous_transcription
                    if len(original_tokens) > 10:
                        previous_transcription = ' '.join(original_tokens[-10:])
                    else:
                        previous_transcription = transcription

                # Keep the last 'overlap_size' bytes for context
                audio_buffer = audio_buffer[chunk_size - overlap_size:]

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stream.stop_stream()

# Start transcription
if __name__ == "__main__":
    transcribe_live()
