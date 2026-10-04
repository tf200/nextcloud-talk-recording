#
# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#

# pylint: disable=missing-docstring

import logging

import numpy
import pytest

from nextcloud.talk.recording import TrackChunker
from nextcloud.talk.recording.TrackChunker import GAP_DURATION, SAMPLE_RATE

def track(duration, talkingRanges, noise=0.0005, level=0.2, seed=1):
    random = numpy.random.default_rng(seed)
    samples = random.normal(0, noise, int(duration * SAMPLE_RATE))
    for start, end in talkingRanges:
        samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)] = random.normal(0, level, int((end - start) * SAMPLE_RATE))
    return (numpy.clip(samples, -1, 1) * 32767).astype(numpy.int16)

def blocks(samples, blockSize=12345):
    return [samples[start:start + blockSize] for start in range(0, len(samples), blockSize)]

class FakeEncoder:

    files = {}

    def __init__(self, fileName):
        self.fileName = fileName
        FakeEncoder.files[fileName] = []

    def write(self, samples):
        FakeEncoder.files[self.fileName].append(numpy.array(samples))

    def close(self):
        pass

class TrackChunkerTest:

    def testDetectSpeechFindsTurnsWithPadding(self):
        samples = track(60, [(5, 8), (30, 31)])

        regions = TrackChunker.detectSpeech(TrackChunker.frameLevels(blocks(samples)))

        assert len(regions) == 2
        assert regions[0][0] == pytest.approx(5 - TrackChunker.PRE_PADDING, abs=0.05)
        assert regions[0][1] == pytest.approx(8 + TrackChunker.POST_PADDING, abs=0.05)
        assert regions[1][0] == pytest.approx(30 - TrackChunker.PRE_PADDING, abs=0.05)

    def testDetectSpeechMergesShortPauses(self):
        samples = track(30, [(5, 8), (9, 12)])

        regions = TrackChunker.detectSpeech(TrackChunker.frameLevels(blocks(samples)))

        assert len(regions) == 1

    def testDetectSpeechAdaptsToNoisyMicrophone(self):
        samples = track(60, [(10, 15)], noise=0.01, level=0.3)

        regions = TrackChunker.detectSpeech(TrackChunker.frameLevels(blocks(samples)))

        assert len(regions) == 1
        assert regions[0][0] == pytest.approx(10 - TrackChunker.PRE_PADDING, abs=0.05)

    def testDetectSpeechInSilentTrack(self):
        samples = numpy.zeros(10 * SAMPLE_RATE, dtype=numpy.int16)

        assert not TrackChunker.detectSpeech(TrackChunker.frameLevels(blocks(samples)))

    def testPlanChunksAddsGapsAndSplitsLongTracks(self):
        chunks = TrackChunker.planChunks([(0, 10), (20, 25), (40, 70)], maxDuration=30)

        assert chunks[0] == [
            {'trackStart': 0, 'duration': 10, 'chunkStart': 0},
            {'trackStart': 20, 'duration': 5, 'chunkStart': 10 + GAP_DURATION},
            {'trackStart': 40, 'duration': 30 - 15 - 2 * GAP_DURATION, 'chunkStart': 15 + 2 * GAP_DURATION},
        ]
        assert chunks[1][0]['trackStart'] == pytest.approx(40 + 30 - 15 - 2 * GAP_DURATION)
        assert chunks[1][0]['chunkStart'] == 0
        for chunk in chunks:
            assert chunk[-1]['chunkStart'] + chunk[-1]['duration'] <= 30

    def testSegmentToRecordingTimeInterpolatesAnchors(self):
        segment = {'recordingOffset': 10, 'anchors': [
            {'segmentTime': 0, 'recordingTime': 10},
            {'segmentTime': 100, 'recordingTime': 110.1},
        ]}

        assert TrackChunker.segmentToRecordingTime(segment, 50) == pytest.approx(60.05)
        assert TrackChunker.segmentToRecordingTime(segment, 200) == pytest.approx(210.1)
        assert TrackChunker.segmentToRecordingTime({'recordingOffset': 3}, 5) == 8

    def testWriteChunksCopiesPiecesWithGaps(self):
        samples = numpy.arange(20 * SAMPLE_RATE, dtype=numpy.int64) % 30000
        chunks = [
            [{'trackStart': 1, 'duration': 2, 'chunkStart': 0}, {'trackStart': 5, 'duration': 1, 'chunkStart': 3}],
            [{'trackStart': 18, 'duration': 3, 'chunkStart': 0}],
        ]
        FakeEncoder.files = {}

        TrackChunker.writeChunks(blocks(samples.astype(numpy.int16)), chunks, ['a', 'b'], FakeEncoder)

        first = numpy.concatenate(FakeEncoder.files['a'])
        assert len(first) == (2 + GAP_DURATION + 1) * SAMPLE_RATE
        assert numpy.array_equal(first[:2 * SAMPLE_RATE], samples[SAMPLE_RATE:3 * SAMPLE_RATE])
        assert not first[2 * SAMPLE_RATE:3 * SAMPLE_RATE].any()
        assert numpy.array_equal(first[3 * SAMPLE_RATE:], samples[5 * SAMPLE_RATE:6 * SAMPLE_RATE])
        # The track ends before the planned end of the last piece.
        second = numpy.concatenate(FakeEncoder.files['b'])
        assert numpy.array_equal(second, samples[18 * SAMPLE_RATE:])

    def testChunkAddsChunksWithRecordingTimes(self, tmp_path):
        samples = track(60, [(5, 8), (30, 31)])
        manifest = {'segments': [{
            'id': 3, 'file': 'segment-3.ogg', 'recordingOffset': 12.5, 'anchors': [],
        }]}
        FakeEncoder.files = {}

        TrackChunker.chunk(manifest, str(tmp_path), logging.getLogger(), lambda fileName: blocks(samples), FakeEncoder)

        assert len(manifest['chunks']) == 1
        chunk = manifest['chunks'][0]
        assert chunk['id'] == '3-0'
        assert chunk['file'] == 'chunk-3-0.ogg'
        assert len(chunk['cutMap']) == 2
        assert chunk['cutMap'][1]['recordingStart'] == pytest.approx(12.5 + 30 - TrackChunker.PRE_PADDING, abs=0.05)
        assert chunk['duration'] == pytest.approx(chunk['cutMap'][1]['chunkStart'] + chunk['cutMap'][1]['duration'])
        assert str(tmp_path / 'chunk-3-0.ogg') in FakeEncoder.files

    def testChunkSkipsFailingSegments(self, tmp_path):
        manifest = {'segments': [{'id': 1, 'file': 'segment-1.ogg'}]}

        def failingDecode(fileName):
            raise RuntimeError('broken')

        TrackChunker.chunk(manifest, str(tmp_path), logging.getLogger(), failingDecode, FakeEncoder)

        assert manifest['chunks'] == []
