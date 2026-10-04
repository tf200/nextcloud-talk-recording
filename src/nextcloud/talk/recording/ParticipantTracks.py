#
# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#

"""
Module to store the per participant audio tracks recorded in the browser.
"""

import base64
import json
import os
import zipfile


class ParticipantTracksWriter:
    """
    Writes the chunks drained from the browser to one file per segment.

    The chunks of a segment are appended in sequence order; chunks received out
    of order are kept in memory until the missing ones arrive.

    The files are written to a directory next to the recording, named after the
    recording file with a ".tracks" suffix.
    """

    def __init__(self, recordingFileName, logger):
        self.directory = recordingFileName + '.tracks'
        self._logger = logger
        self._extension = 'ogg'
        self._nextSeq = {}
        self._pendingChunks = {}

        os.makedirs(self.directory, exist_ok=True)

    def setMimeType(self, mimeType):
        """
        Sets the extension of the segment files from the MIME type of the
        chunks.

        :param mimeType: the MIME type used by the browser recorder.
        """

        self._extension = 'webm' if mimeType and 'webm' in mimeType else 'ogg'

    def write(self, chunks):
        """
        Appends the given chunks to their segment files.

        :param chunks: list of dicts with "segmentId", "seq" and "data" (base64).
        """

        for chunk in chunks or []:
            segmentId = str(chunk['segmentId'])
            self._pendingChunks.setdefault(segmentId, {})[int(chunk['seq'])] = base64.b64decode(chunk['data'])

            self._flush(segmentId)

    def _flush(self, segmentId):
        pending = self._pendingChunks[segmentId]
        nextSeq = self._nextSeq.get(segmentId, 0)

        if nextSeq not in pending:
            return

        with open(self.getSegmentFileName(segmentId), 'ab') as segmentFile:
            while nextSeq in pending:
                segmentFile.write(pending.pop(nextSeq))
                nextSeq += 1

        self._nextSeq[segmentId] = nextSeq

    def getSegmentFileName(self, segmentId):
        """
        :param segmentId: the id of the segment.
        :return: the path of the file of the segment.
        """

        if not str(segmentId).isdigit():
            raise ValueError(f"Invalid segment id: {segmentId}")

        return os.path.join(self.directory, f'segment-{segmentId}.{self._extension}')

    def writeManifest(self, manifest, recordingOffset, clockUncertainty):
        """
        Writes the manifest of the recorded tracks.

        Besides the manifest returned by the browser the manifest includes the
        file name of each segment and the estimated offset between the start of
        the recorder and the start of the track recording (in seconds; the page
        time of the start of the recording file is
        "startPageTime - recordingOffset").

        :param manifest: the manifest returned by the browser.
        :param recordingOffset: seconds between starting the recorder and
               starting the track recording.
        :param clockUncertainty: uncertainty of recordingOffset in seconds.
        :return: the path of the manifest file.
        """

        for segmentId, pending in self._pendingChunks.items():
            if pending:
                self._logger.warning("Segment %s is missing chunk %s; %d later chunks were dropped", segmentId, self._nextSeq.get(segmentId, 0), len(pending))

        segments = []
        for segment in manifest.get('segments', []):
            segmentFileName = self.getSegmentFileName(segment['id'])
            if not os.path.exists(segmentFileName):
                continue

            segments.append({
                **segment,
                'file': os.path.basename(segmentFileName),
            })

        manifestFileName = os.path.join(self.directory, 'manifest.json')
        with open(manifestFileName, 'w', encoding='utf-8') as manifestFile:
            json.dump({
                **manifest,
                'segments': segments,
                'recordingOffset': recordingOffset,
                'clockUncertainty': clockUncertainty,
            }, manifestFile)

        return manifestFileName


def writeArchive(directory, manifest, archiveFileName):
    """
    Writes the manifest and the chunk files to a zip archive to be uploaded.

    The segment files are not included, as only the chunks are transcribed. The
    chunks are already compressed, so they are just stored.

    :param directory: the directory of the participant tracks.
    :param manifest: the manifest with the chunks added by TrackChunker.
    :param archiveFileName: the path of the archive to write.
    :return: the path of the archive.
    """

    with zipfile.ZipFile(archiveFileName, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        for chunk in manifest.get('chunks', []):
            archive.write(os.path.join(directory, chunk['file']), chunk['file'])

    return archiveFileName
