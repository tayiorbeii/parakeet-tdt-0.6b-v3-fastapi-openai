import os
import subprocess
import wave


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds.

    Prefer reading WAV metadata directly because the app converts uploads to WAV
    before probing. Fall back to ffprobe for non-WAV inputs used by benchmarks.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".wav", ".wave"}:
        try:
            with wave.open(file_path, "rb") as audio_file:
                frame_rate = audio_file.getframerate()
                if frame_rate <= 0:
                    return 0.0
                return audio_file.getnframes() / float(frame_rate)
        except (wave.Error, OSError, ValueError, ZeroDivisionError) as e:
            print(f"Could not get duration of WAV file '{file_path}': {e}")
            return 0.0

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        print(f"Could not get duration of file '{file_path}': {e}")
        return 0.0
