import collections

from webrtcvad import Vad


def webrtc_vad(audio, rate, aggressiveness=3, frame_duration_ms=30, window_duration_ms=300):
    vad = Vad(aggressiveness)
    num_window_frames = int(window_duration_ms / frame_duration_ms)
    sliding_window = collections.deque(maxlen=num_window_frames)
    triggered = False

    voiced_frames = []
    for frame in generate_frames(audio, rate, frame_duration_ms):
        is_speech = vad.is_speech(frame.bytes, rate)
        sliding_window.append((frame, is_speech))

        if not triggered:
            num_voiced = len([f for f, speech in sliding_window if speech])
            if num_voiced > 0.9 * sliding_window.maxlen:
                triggered = True
                voiced_frames += [frame for frame, _ in sliding_window]
                sliding_window.clear()
        else:
            voiced_frames.append(frame)
            num_unvoiced = len([f for f, speech in sliding_window if not speech])
            if num_unvoiced > 0.9 * sliding_window.maxlen:
                triggered = False
                yield voiced_frames, rate
                sliding_window.clear()
                voiced_frames = []
    if voiced_frames:
        yield voiced_frames, rate


def generate_frames(audio, sample_rate, frame_duration_ms=30):
    frame_length = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    timestamp = 0.0
    duration = (float(frame_length) / sample_rate) / 2.0
    while offset + frame_length < len(audio):
        yield Frame(audio[offset:offset + frame_length], timestamp, duration)
        timestamp += duration
        offset += frame_length
