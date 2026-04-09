import re
from typing import List


def parse_srt_timecode(timecode: str) -> float:
    hours, minutes, seconds_ms = timecode.split(":")
    seconds, milliseconds = seconds_ms.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000.0
    )


def format_compact_timecode(timecode: str) -> str:
    hours, minutes, seconds_ms = timecode.split(":")
    seconds = seconds_ms.split(",")[0]
    if hours == "00":
        return f"{minutes}:{seconds}"
    return f"{hours}:{minutes}:{seconds}"


def parse_srt_entries(srt_text: str) -> List[dict]:
    entries = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())

    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        time_line = lines[1] if "-->" in lines[1] else lines[0]
        if "-->" not in time_line:
            continue

        start_time, end_time = [part.strip() for part in time_line.split("-->", 1)]
        text_lines = lines[2:] if time_line == lines[1] else lines[1:]
        text = " ".join(text_lines).strip()
        if not text:
            continue

        entries.append(
            {
                "startTime": start_time,
                "endTime": end_time,
                "startSeconds": parse_srt_timecode(start_time),
                "endSeconds": parse_srt_timecode(end_time),
                "text": text,
            }
        )

    return entries


def srt_to_readable_text(srt_text: str, time_limit_in_seconds: float = 20.0) -> str:
    entries = parse_srt_entries(srt_text)
    if not entries:
        return ""

    with_times = [
        {**line, "totalSeconds": line["endSeconds"] - line["startSeconds"]}
        for line in entries
    ]

    array_by_times = []
    temp_array = []
    current_time_in_seconds = 0.0

    for x in with_times:
        if current_time_in_seconds + x["totalSeconds"] >= time_limit_in_seconds:
            if temp_array:
                array_by_times.append(temp_array)
            temp_array = []
            current_time_in_seconds = 0.0

        if current_time_in_seconds == 0:
            temp_array.append(f"[{format_compact_timecode(x['startTime'])}] {x['text']}")
        else:
            temp_array.append(x["text"])

        current_time_in_seconds += x["totalSeconds"]

    if temp_array:
        array_by_times.append(temp_array)

    return "\n\n".join(" ".join(chunk) for chunk in array_by_times if chunk)
