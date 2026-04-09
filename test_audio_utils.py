import os
import tempfile
import wave
from pathlib import Path

from audio_utils import get_audio_duration
from app import get_direct_output_paths
from transcript_utils import srt_to_readable_text


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


def test_srt_to_readable_text_groups_cues_like_the_kit_script():
    srt_text = """1
00:00:01,000 --> 00:00:03,000
Hello

2
00:00:03,000 --> 00:00:06,000
world

3
00:00:06,000 --> 00:00:10,000
again

4
00:00:10,000 --> 00:00:31,000
next block
"""

    expected = """[00:01] Hello world again

[00:10] next block"""

    assert srt_to_readable_text(srt_text) == expected


def test_direct_output_paths_stay_next_to_source_file():
    source = Path("/tmp/videos/demo clip.mp4")
    srt_path, txt_path = get_direct_output_paths(source)

    assert srt_path == Path("/tmp/videos/demo clip.srt")
    assert txt_path == Path("/tmp/videos/demo clip-transcript.txt")
