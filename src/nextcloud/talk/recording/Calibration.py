#
# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#

"""
Module to align the participant tracks with the recording file.

The recording file contains the mix of all the participant tracks, so the
position of a track in the recording can be found by cross-correlating the
track with the recording. The amplitude envelopes are correlated rather than the
waveforms, as both are lossy encoded independently and the waveforms are not
preserved, but the envelope of speech is.

Each segment is aligned at several points ("anchors") spread over the segment,
which also allows to compensate any drift between the clocks. Segments that can
not be aligned (for example, if the participant never spoke) keep the offset
estimated from the clocks.
"""

import subprocess

import numpy

SAMPLE_RATE = 8000
ENVELOPE_RATE = 200
PROBE_DURATION = 20
SEARCH_MARGIN = 5
MIN_SCORE = 0.3
SECONDS_PER_ANCHOR = 300
MAX_ANCHORS_PER_SEGMENT = 12


def decodeAudio(fileName, start=0, duration=None):
    """
    Decodes the given audio (or video) file to mono PCM.

    :param fileName: the file to decode.
    :param start: the position to start decoding from, in seconds.
    :param duration: the duration to decode, in seconds, or None for all.
    :return: numpy float32 array with the samples at SAMPLE_RATE.
    """

    arguments = ['ffmpeg', '-nostdin', '-loglevel', 'error']
    if start > 0:
        arguments += ['-ss', f'{start:.3f}']
    arguments += ['-i', fileName]
    if duration is not None:
        arguments += ['-t', f'{duration:.3f}']
    arguments += ['-vn', '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 's16le', '-']

    result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    return numpy.frombuffer(result.stdout, dtype=numpy.int16).astype(numpy.float32) / 32768


def envelope(samples):
    """
    :param samples: samples at SAMPLE_RATE.
    :return: the amplitude envelope at ENVELOPE_RATE.
    """

    blockSize = SAMPLE_RATE // ENVELOPE_RATE
    blocks = len(samples) // blockSize

    return numpy.abs(samples[:blocks * blockSize]).reshape(blocks, blockSize).mean(axis=1)


def findOffset(reference, probe):
    """
    Finds the position of the probe in the reference.

    Both arguments are envelopes at ENVELOPE_RATE, and the reference must be
    longer than the probe. The score is the normalized correlation at the best
    position (1 is a perfect match).

    :return: tuple with the position in seconds and the score.
    """

    probe = probe - probe.mean()
    probeNorm = numpy.linalg.norm(probe)
    positions = len(reference) - len(probe) + 1
    if probeNorm == 0 or positions <= 0:
        return None, 0

    size = 1 << int(numpy.ceil(numpy.log2(len(reference) + len(probe))))
    correlation = numpy.fft.irfft(numpy.fft.rfft(reference, size) * numpy.conj(numpy.fft.rfft(probe, size)), size)[:positions]

    # Normalize by the energy of the reference under the probe at each
    # position, with the mean of that window removed.
    cumulative = numpy.concatenate(([0], numpy.cumsum(reference)))
    cumulativeSquares = numpy.concatenate(([0], numpy.cumsum(reference * reference)))
    windowSum = cumulative[len(probe):] - cumulative[:positions]
    windowSquares = cumulativeSquares[len(probe):] - cumulativeSquares[:positions]
    windowVariance = numpy.maximum(windowSquares - windowSum * windowSum / len(probe), 1e-12)

    scores = correlation / (probeNorm * numpy.sqrt(windowVariance))
    best = int(numpy.argmax(scores))

    return best / ENVELOPE_RATE, float(scores[best])


def selectProbeStarts(segmentEnvelope):
    """
    Selects the start of the probe windows of a segment.

    The segment is split in parts of SECONDS_PER_ANCHOR, and in each part the
    window with the highest envelope variance (that is, with most speech
    activity) is selected.

    :return: list of window start positions in seconds.
    """

    probeLength = PROBE_DURATION * ENVELOPE_RATE
    if len(segmentEnvelope) < probeLength:
        return [0] if len(segmentEnvelope) >= ENVELOPE_RATE else []

    parts = min(MAX_ANCHORS_PER_SEGMENT, max(1, round(len(segmentEnvelope) / (SECONDS_PER_ANCHOR * ENVELOPE_RATE))))
    partLength = len(segmentEnvelope) // parts
    step = ENVELOPE_RATE

    starts = []
    for part in range(parts):
        partStart = part * partLength
        partEnd = min(len(segmentEnvelope), partStart + partLength + probeLength)
        candidates = range(partStart, partEnd - probeLength + 1, step)
        if not candidates:
            continue

        best = max(candidates, key=lambda start: segmentEnvelope[start:start + probeLength].var())
        starts.append(best / ENVELOPE_RATE)

    return starts


def calibrateSegment(recordingFileName, segmentFileName, estimatedOffset, decode=decodeAudio):
    """
    Aligns a segment with the recording.

    :param recordingFileName: the recording file (the mix).
    :param segmentFileName: the file of the segment.
    :param estimatedOffset: the estimated position of the segment start in the
           recording, in seconds.
    :return: list of anchors {"segmentTime", "recordingTime", "score"}; empty
             if the segment could not be aligned.
    """

    segmentEnvelope = envelope(decode(segmentFileName))

    anchors = []
    for probeStart in selectProbeStarts(segmentEnvelope):
        probe = segmentEnvelope[int(probeStart * ENVELOPE_RATE):int((probeStart + PROBE_DURATION) * ENVELOPE_RATE)]

        searchStart = max(0, estimatedOffset + probeStart - SEARCH_MARGIN)
        searchDuration = len(probe) / ENVELOPE_RATE + estimatedOffset + probeStart + SEARCH_MARGIN - searchStart
        reference = envelope(decode(recordingFileName, searchStart, searchDuration))

        position, score = findOffset(reference, probe)
        if position is None or score < MIN_SCORE:
            continue

        anchors.append({
            'segmentTime': probeStart,
            'recordingTime': round(searchStart + position, 3),
            'score': round(score, 3),
        })

    return anchors


def calibrate(recordingFileName, manifest, tracksDirectory, logger, decode=decodeAudio):
    """
    Adds the alignment with the recording to each segment of the manifest.

    Each segment gets a "recordingOffset" (position of the segment start in the
    recording, in seconds), the "alignment" method used ("xcorr" or "clock")
    and the "anchors" found.

    :param manifest: the manifest written by ParticipantTracksWriter.
    :return: the manifest with the alignment.
    """

    fileZeroPageTime = manifest['startPageTime'] - manifest.get('recordingOffset', 0)

    for segment in manifest.get('segments', []):
        estimatedOffset = max(0, segment['startPageTime'] - fileZeroPageTime) if segment.get('startPageTime') is not None else 0

        try:
            anchors = calibrateSegment(recordingFileName, f"{tracksDirectory}/{segment['file']}", estimatedOffset, decode)
        except Exception:
            logger.exception("Error when aligning segment %s", segment.get('id'))
            anchors = []

        segment['anchors'] = anchors
        if anchors:
            segment['alignment'] = 'xcorr'
            segment['recordingOffset'] = round(float(numpy.median([anchor['recordingTime'] - anchor['segmentTime'] for anchor in anchors])), 3)
        else:
            segment['alignment'] = 'clock'
            segment['recordingOffset'] = round(estimatedOffset, 3)

    return manifest
