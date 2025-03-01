from pydub import AudioSegment
import os

def split_wav(file_path, interval_minutes=10):
    """Splits a .wav file into 10-minute intervals."""
    
    audio = AudioSegment.from_wav(file_path)
    interval_ms = interval_minutes * 60 * 1000  # Convert minutes to milliseconds
    total_length = len(audio)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)

    for i, start_time in enumerate(range(0, total_length, interval_ms)):
        chunk = audio[start_time:start_time + interval_ms]
        chunk_name = f"{base_name}_part_{i+1}.wav"
        chunk_path = os.path.join(output_dir, chunk_name)
        chunk.export(chunk_path, format="wav")
        print(f"Exported: {chunk_path}")

if __name__ == "__main__":
    file_path = "zoom.wav"  # Replace with your actual file path
    split_wav(file_path)
