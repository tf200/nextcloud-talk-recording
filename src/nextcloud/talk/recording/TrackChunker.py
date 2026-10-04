#
# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#

"""
Module to split the participant tracks in speech-only chunks to be transcribed.

Each track contains a single participant, who is silent most of the time, so
only the parts with speech are kept. They are joined in chunks with a short
silence between them; the chunks are limited in duration, as transcription
providers limit the duration of the audio when word timestamps are requested.

Every chunk has a cut map with the position in the chunk and in the recording of
each of its pieces, so the time of a transcribed word can be converted to the
time in the recording.

The tracks are decoded as a stream, so the memory used does not depend on the
duration of the call.
"""

import os
import subprocess

import numpy

SAMPLE_RATE = 16000
FRAME_DURATION = 0.02
BLOCK_DURATION = 10

# Speech detection.
THRESHOLD_ABOVE_NOISE_FLOOR = 12
MIN_THRESHOLD = -50
MAX_THRESHOLD = -30
SILENCE_LEVEL = -100
MIN_SPEECH_DURATION = 0.1
PRE_PADDING = 0.4
POST_PADDING = 0.6
MERGE_GAP = 1.5

# Chunks.
GAP_DURATION = 1.0
# Gemini Transcribe accepts up to 15 minutes per request with word timestamps.
MAX_CHUNK_DURATION = 14 * 60


def openDecoder(fileName):
    """
    :return: ffmpeg process writing the file as 16 bit mono PCM at SAMPLE_RATE
             to its stdout.
    """

    return subprocess.Popen(
        ['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', fileName, '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 's16le', '-'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def decodeBlocks(fileName):
    """
    Decodes the given file.

    :return: generator of numpy int16 arrays with consecutive samples.
    """

    process = openDecoder(fileName)
    blockBytes = BLOCK_DURATION * SAMPLE_RATE * 2
    try:
        while True:
            data = process.stdout.read(blockBytes)
            if not data:
                break

            # An odd number of bytes can only happen at the end of the stream.
            yield numpy.frombuffer(data[:len(data) - len(data) % 2], dtype=numpy.int16)
    finally:
        process.stdout.close()
        process.wait()


def frameLevels(blocks):
    """
    :param blocks: iterable of int16 sample arrays.
    :return: numpy array with the level (dBFS) of each frame of FRAME_DURATION.
    """

    frameSize = int(FRAME_DURATION * SAMPLE_RATE)
    levels = []
    leftover = numpy.zeros(0, dtype=numpy.float32)
    for block in blocks:
        samples = numpy.concatenate((leftover, block.astype(numpy.float32) / 32768))
        frames = len(samples) // frameSize
        if frames:
            power = numpy.mean(numpy.square(samples[:frames * frameSize].reshape(frames, frameSize)), axis=1)
            levels.append(10 * numpy.log10(numpy.maximum(power, 10 ** (SILENCE_LEVEL / 10))))
        leftover = samples[frames * frameSize:]

    return numpy.concatenate(levels) if levels else numpy.zeros(0)


def detectSpeech(levels):
    """
    Finds the parts with speech.

    The threshold is relative to the noise floor of the track, so it adapts to
    the microphone and the background noise of each participant.

    :param levels: frame levels as returned by frameLevels().
    :return: list of (start, end) tuples in seconds, padded and merged.
    """

    audible = levels[levels > SILENCE_LEVEL + 10]
    if len(audible) == 0:
        return []

    noiseFloor = numpy.percentile(audible, 10)
    threshold = min(MAX_THRESHOLD, max(MIN_THRESHOLD, noiseFloor + THRESHOLD_ABOVE_NOISE_FLOOR))

    speech = numpy.concatenate(([False], levels > threshold, [False]))
    changes = numpy.flatnonzero(speech[1:] != speech[:-1])
    duration = len(levels) * FRAME_DURATION

    regions = []
    for startFrame, endFrame in zip(changes[0::2], changes[1::2]):
        if (endFrame - startFrame) * FRAME_DURATION < MIN_SPEECH_DURATION:
            continue

        start = max(0, startFrame * FRAME_DURATION - PRE_PADDING)
        end = min(duration, endFrame * FRAME_DURATION + POST_PADDING)
        if regions and start - regions[-1][1] <= MERGE_GAP:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))

    return [(round(float(start), 3), round(float(end), 3)) for start, end in regions]


def planChunks(regions, maxDuration=MAX_CHUNK_DURATION):
    """
    Groups the speech regions in chunks.

    :param regions: list of (start, end) tuples in seconds of track time.
    :return: list of chunks; each chunk is a list of pieces
             {"trackStart", "duration", "chunkStart"}.
    """

    chunks = []
    pieces = []
    chunkDuration = 0
    for start, end in regions:
        while end - start > 0:
            gap = GAP_DURATION if pieces else 0
            available = maxDuration - chunkDuration - gap
            if available <= 0:
                chunks.append(pieces)
                pieces = []
                chunkDuration = 0
                continue

            duration = min(end - start, available)
            pieces.append({
                'trackStart': round(start, 3),
                'duration': round(duration, 3),
                'chunkStart': round(chunkDuration + gap, 3),
            })
            chunkDuration += gap + duration
            start += duration

    if pieces:
        chunks.append(pieces)

    return chunks


def segmentToRecordingTime(segment, trackTime):
    """
    Converts a time of a segment to the time in the recording.

    Between anchors the offset is linearly interpolated, which compensates any
    drift between the clocks; outside them the nearest anchor is used.

    :param segment: manifest segment with the alignment added by Calibration.
    :param trackTime: seconds since the start of the segment.
    :return: seconds since the start of the recording.
    """

    anchors = sorted(segment.get('anchors') or [], key=lambda anchor: anchor['segmentTime'])
    if not anchors:
        return segment.get('recordingOffset', 0) + trackTime

    segmentTimes = [anchor['segmentTime'] for anchor in anchors]
    offsets = [anchor['recordingTime'] - anchor['segmentTime'] for anchor in anchors]

    return float(numpy.interp(trackTime, segmentTimes, offsets)) + trackTime


class ChunkEncoder:
    """
    Encodes samples as Ogg Opus in a file.
    """

    def __init__(self, fileName):
        # pylint: disable=consider-using-with
        self._process = subprocess.Popen(
            ['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1', '-i', '-',
             '-c:a', 'libopus', '-b:a', '32k', '-application', 'voip', fileName],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, samples):
        self._process.stdin.write(samples.astype(numpy.int16).tobytes())

    def close(self):
        self._process.stdin.close()
        if self._process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {self._process.returncode}")


def writeChunks(blocks, chunks, fileNames, encoderFactory=ChunkEncoder):
    """
    Writes the pieces of each chunk, separated by silence, to its file.

    :param blocks: iterable of int16 sample arrays of the whole track.
    :param chunks: chunks as returned by planChunks().
    :param fileNames: the file name of each chunk.
    """

    pieces = []
    for chunkIndex, chunk in enumerate(chunks):
        for pieceIndex, piece in enumerate(chunk):
            start = round(piece['trackStart'] * SAMPLE_RATE)
            pieces.append((chunkIndex, pieceIndex, start, start + round(piece['duration'] * SAMPLE_RATE)))

    encoder = None
    encoderChunk = None
    current = 0
    started = False

    def startPiece():
        nonlocal encoder, encoderChunk, started
        chunkIndex, pieceIndex, _, _ = pieces[current]
        if chunkIndex != encoderChunk:
            if encoder:
                encoder.close()
            encoder = encoderFactory(fileNames[chunkIndex])
            encoderChunk = chunkIndex
        if pieceIndex > 0:
            encoder.write(numpy.zeros(round(GAP_DURATION * SAMPLE_RATE), dtype=numpy.int16))
        started = True

    try:
        position = 0
        for block in blocks:
            blockEnd = position + len(block)
            while current < len(pieces) and pieces[current][2] < blockEnd:
                if not started:
                    startPiece()

                _, _, start, end = pieces[current]
                if max(start, position) < min(end, blockEnd):
                    encoder.write(block[max(start, position) - position:min(end, blockEnd) - position])
                if end > blockEnd:
                    break

                current += 1
                started = False
            position = blockEnd

        # Pieces after the end of the decoded audio still get their (empty)
        # file, so every planned chunk exists.
        while current < len(pieces):
            if not started:
                startPiece()
            current += 1
            started = False
    finally:
        if encoder:
            encoder.close()


def chunkSegment(segment, tracksDirectory, decode=decodeBlocks, encoderFactory=ChunkEncoder):
    """
    Splits a segment in speech-only chunks.

    :param segment: manifest segment with the alignment added by Calibration.
    :return: list of chunk manifest entries.
    """

    segmentFileName = os.path.join(tracksDirectory, segment['file'])
    chunks = planChunks(detectSpeech(frameLevels(decode(segmentFileName))))

    entries = []
    for index, pieces in enumerate(chunks):
        entries.append({
            'id': f"{segment['id']}-{index}",
            'segmentId': segment['id'],
            'file': f"chunk-{segment['id']}-{index}.ogg",
            'mimeType': 'audio/ogg',
            'duration': round(pieces[-1]['chunkStart'] + pieces[-1]['duration'], 3),
            'cutMap': [{
                'chunkStart': piece['chunkStart'],
                'trackStart': piece['trackStart'],
                'recordingStart': round(segmentToRecordingTime(segment, piece['trackStart']), 3),
                'duration': piece['duration'],
            } for piece in pieces],
        })

    if entries:
        writeChunks(decode(segmentFileName), chunks, [os.path.join(tracksDirectory, entry['file']) for entry in entries], encoderFactory)

    return entries


def chunk(manifest, tracksDirectory, logger, decode=decodeBlocks, encoderFactory=ChunkEncoder):
    """
    Adds the speech-only chunks of all the segments to the manifest.

    Segments that fail are logged and skipped, so the rest can still be
    transcribed.

    :return: the manifest with "chunks".
    """

    manifest['chunks'] = []
    for segment in manifest.get('segments', []):
        try:
            manifest['chunks'] += chunkSegment(segment, tracksDirectory, decode, encoderFactory)
        except Exception:
            logger.exception("Error when chunking segment %s", segment.get('id'))

    return manifest
