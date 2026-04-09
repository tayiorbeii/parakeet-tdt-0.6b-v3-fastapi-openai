import os
import tempfile
import wave

from audio_utils import get_audio_duration


def test_get_audio_duration_reads_wav_header_without_ffprobe():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        sample_rate = 16000
        frame_count = sample_rate * 2
        with wave.open(wav_path, "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(sample_rate)
            audio_file.writeframes(b"\x00\x00" * frame_count)

        duration = get_audio_duration(wav_path)

        assert abs(duration - 2.0) < 1e-6
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
