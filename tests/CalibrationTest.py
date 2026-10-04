#
# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#

# pylint: disable=missing-docstring

import logging

import numpy
import pytest

from nextcloud.talk.recording import Calibration
from nextcloud.talk.recording.Calibration import SAMPLE_RATE

def speechLike(random, duration, talkingRanges):
    """
    Noise bursts with syllable-like on/off modulation inside the talking ranges.
    """

    samples = numpy.zeros(int(duration * SAMPLE_RATE), dtype=numpy.float32)
    for talkingStart, talkingEnd in talkingRanges:
        time = talkingStart
        while time < talkingEnd:
            syllable = random.uniform(0.08, 0.35)
            start = int(time * SAMPLE_RATE)
            end = min(int((time + syllable) * SAMPLE_RATE), int(talkingEnd * SAMPLE_RATE))
            samples[start:end] = random.normal(0, random.uniform(0.1, 0.4), end - start)
            time += syllable + random.uniform(0.03, 0.4)
    return samples

def place(target, samples, offset):
    start = int(offset * SAMPLE_RATE)
    end = min(len(target), start + len(samples))
    target[start:end] += samples[:end - start]

class CalibrationTest:

    @pytest.fixture
    def random(self):
        return numpy.random.default_rng(42)

    def decoder(self, files):
        def decode(fileName, start=0, duration=None):
            samples = files[fileName]
            first = int(start * SAMPLE_RATE)
            last = len(samples) if duration is None else first + int(duration * SAMPLE_RATE)
            return samples[first:last]
        return decode

    def testFindOffset(self, random):
        reference = Calibration.envelope(speechLike(random, 60, [(0, 60)]))
        probe = reference[int(23.4 * 200):int(43.4 * 200)]

        position, score = Calibration.findOffset(reference, probe)

        assert position == pytest.approx(23.4, abs=0.005)
        assert score > 0.99

    def testFindOffsetWithoutSignal(self):
        position, score = Calibration.findOffset(numpy.ones(1000), numpy.zeros(100))

        assert position is None
        assert score == 0

    def testCalibrateCorrectsWrongClockEstimate(self, random):
        # Alice talks most of the time; Bob joins later and interrupts her.
        alice = speechLike(random, 300, [(1, 60), (75, 140), (150, 290)])
        bob = speechLike(random, 200, [(5, 15), (40, 60), (120, 180)])

        # Real positions in the recording; the clocks are off by 1.2 seconds
        # (for example, due to the recorder start up time).
        aliceOffset = 2.35
        bobOffset = 91.8
        recording = numpy.zeros(int(400 * SAMPLE_RATE), dtype=numpy.float32)
        place(recording, alice * 0.8, aliceOffset)
        place(recording, bob * 1.1, bobOffset)
        # Lossy encoding of both files is approximated with independent noise.
        recording += random.normal(0, 0.01, len(recording)).astype(numpy.float32)

        decode = self.decoder({
            'recording.webm': recording,
            'tracks/segment-1.ogg': alice,
            'tracks/segment-2.ogg': bob,
            'tracks/segment-3.ogg': numpy.zeros(10 * SAMPLE_RATE, dtype=numpy.float32),
        })
        manifest = {
            'startPageTime': 100,
            'recordingOffset': 0.5,
            'segments': [
                {'id': '1', 'file': 'segment-1.ogg', 'startPageTime': 99.5 + aliceOffset - 1.2},
                {'id': '2', 'file': 'segment-2.ogg', 'startPageTime': 99.5 + bobOffset - 1.2},
                {'id': '3', 'file': 'segment-3.ogg', 'startPageTime': 150},
            ],
        }

        Calibration.calibrate('recording.webm', manifest, 'tracks', logging.getLogger(__name__), decode)

        alignedAlice, alignedBob, silent = manifest['segments']
        assert alignedAlice['alignment'] == 'xcorr'
        assert alignedAlice['recordingOffset'] == pytest.approx(aliceOffset, abs=0.01)
        assert len(alignedAlice['anchors']) == 1
        assert alignedBob['alignment'] == 'xcorr'
        assert alignedBob['recordingOffset'] == pytest.approx(bobOffset, abs=0.01)
        assert silent['alignment'] == 'clock'
        assert silent['recordingOffset'] == pytest.approx(50.5)
        assert silent['anchors'] == []

    def testSeveralAnchorsForLongSegments(self, random):
        track = speechLike(random, 1200, [(0, 1200)])
        recording = numpy.zeros(int(1210 * SAMPLE_RATE), dtype=numpy.float32)
        place(recording, track, 4)

        anchors = Calibration.calibrateSegment('recording', 'track', 3, self.decoder({'recording': recording, 'track': track}))

        assert len(anchors) == 4
        for anchor in anchors:
            assert anchor['recordingTime'] - anchor['segmentTime'] == pytest.approx(4, abs=0.01)
