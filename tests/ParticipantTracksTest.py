#
# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#

# pylint: disable=missing-docstring

import base64
import json
import logging
import os
import zipfile

import pytest

from nextcloud.talk.recording.ParticipantTracks import ParticipantTracksWriter, writeArchive

def chunk(segmentId, seq, data):
    return {
        'segmentId': segmentId,
        'seq': seq,
        'data': base64.b64encode(data).decode('ascii'),
    }

class ParticipantTracksWriterTest:

    @pytest.fixture
    def writer(self, tmp_path):
        writer = ParticipantTracksWriter(str(tmp_path / 'Recording.webm'), logging.getLogger(__name__))
        writer.setMimeType('audio/webm;codecs=opus')
        return writer

    def testWriteChunksInSequenceOrder(self, writer):
        writer.write([chunk('1', 1, b'b'), chunk('2', 0, b'x')])
        writer.write([chunk('1', 0, b'a'), chunk('1', 2, b'c')])

        with open(writer.getSegmentFileName('1'), 'rb') as segmentFile:
            assert segmentFile.read() == b'abc'
        with open(writer.getSegmentFileName('2'), 'rb') as segmentFile:
            assert segmentFile.read() == b'x'

    def testExtensionFromMimeType(self, writer):
        assert writer.getSegmentFileName('1').endswith('segment-1.webm')

        writer.setMimeType('audio/ogg;codecs=opus')

        assert writer.getSegmentFileName('1').endswith('segment-1.ogg')

    def testInvalidSegmentId(self, writer):
        with pytest.raises(ValueError):
            writer.getSegmentFileName('../escape')

    def testWriteManifest(self, writer):
        writer.write([chunk('1', 0, b'a'), chunk('2', 1, b'never flushed')])

        manifestFileName = writer.writeManifest({
            'version': 1,
            'startPageTime': 12.5,
            'segments': [
                {'id': '1', 'displayName': 'Alice'},
                {'id': '2', 'displayName': 'Bob'},
            ],
        }, 0.75, 0.01)

        with open(manifestFileName, encoding='utf-8') as manifestFile:
            manifest = json.load(manifestFile)

        assert os.path.dirname(manifestFileName) == writer.directory
        assert manifest == {
            'version': 1,
            'startPageTime': 12.5,
            'segments': [
                {'id': '1', 'displayName': 'Alice', 'file': 'segment-1.webm'},
            ],
            'recordingOffset': 0.75,
            'clockUncertainty': 0.01,
        }

    def testWriteArchiveContainsManifestAndChunksOnly(self, tmp_path):
        (tmp_path / 'segment-1.ogg').write_bytes(b'segment')
        (tmp_path / 'chunk-1-0.ogg').write_bytes(b'chunk')
        manifest = {'segments': [{'id': 1, 'file': 'segment-1.ogg'}], 'chunks': [{'id': '1-0', 'file': 'chunk-1-0.ogg'}]}

        archiveFileName = writeArchive(str(tmp_path), manifest, str(tmp_path / 'tracks.zip'))

        with zipfile.ZipFile(archiveFileName) as archive:
            assert sorted(archive.namelist()) == ['chunk-1-0.ogg', 'manifest.json']
            assert json.loads(archive.read('manifest.json')) == manifest
            assert archive.read('chunk-1-0.ogg') == b'chunk'
