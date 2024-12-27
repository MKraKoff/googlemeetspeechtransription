# MODEL: Specifies the Whisper model to use for transcription.
# Available options include "tiny", "base", "small", "medium", and "large".
# Larger models provide higher accuracy but require more computational resources.
MODEL = "base"

# BUFFER_DURATION: Duration (in seconds) of audio to process in each transcription cycle.
# A higher value reduces the frequency of transcription but may introduce latency.
# A lower value processes audio more frequently but may be less efficient.
BUFFER_DURATION = 6

# OVERLAP_DURATION: Duration (in seconds) of overlapping audio between consecutive transcription cycles.
# Overlap helps maintain context between chunks, improving the continuity of transcription.
# Typically, 1-2 seconds is sufficient to handle transitions smoothly.
OVERLAP_DURATION = 1
