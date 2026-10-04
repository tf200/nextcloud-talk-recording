#
# SPDX-FileCopyrightText: 2023 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#

"""
Module to start and stop the recording for a specific call.
"""

import logging
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from secrets import token_urlsafe
from threading import Event, Thread

import pulsectl
from pyvirtualdisplay import Display

from . import BackendNotifier, Calibration, TrackChunker
from .Config import config
from .Participant import Participant
from .ParticipantTracks import ParticipantTracksWriter, writeArchive
from .RecorderArgumentsBuilder import RecorderArgumentsBuilder

def newAudioSink(baseSinkName):
    """
    Start new audio sink for the audio output of the browser.

    Each browser instance uses its own sink that will then be captured by the
    recorder (using the monitor of that sink as the source). Otherwise several
    browsers would use the same default sink, and their audio output would be
    mixed.

    The sink is created by loading a null sink module. This module needs to be
    unloaded once the sink is no longer needed to remove it (which also removes
    the monitor of that sink).

    :param baseSinkName: the base name for the sink; it is expected to have been
           sanitized and to contain only alpha-numeric characters.
    :return: a tuple with the module index, the sink index and the source index
             (the monitor of the sink), all as ints.
    """

    # A random value is appended to the base sink name to "ensure" that there
    # will be no name clashes if a previous sink with that base name was not
    # unloaded yet.
    sinkName = f"{baseSinkName}-{token_urlsafe(32)}"

    # Module names can be, at most, 127 characters, so the name is truncated if
    # needed.
    sinkName = sinkName[:127]

    with pulsectl.Pulse(f"{sinkName}-loader") as pacmd:
        pacmd.module_load("module-null-sink", f"sink_name={sinkName}")

        moduleIndex = None
        moduleList = pacmd.module_list()
        for module in moduleList:
            if module.argument == f"sink_name={sinkName}":
                moduleIndex = module.index

        if not moduleIndex:
            raise Exception(f"New audio module for sink {sinkName} not found ({moduleList})")

        sinkIndex = None
        sinkList = pacmd.sink_list()
        for sink in sinkList:
            if sink.name == sinkName:
                sinkIndex = sink.index

        if not sinkIndex:
            raise Exception(f"New audio sink {sinkName} not found ({sinkList})")

        sourceIndex = None
        sourceList = pacmd.source_list()
        for source in sourceList:
            if source.monitor_of_sink == sinkIndex:
                sourceIndex = source.index

        if not sourceIndex:
            raise Exception(f"Audio source (monitor) of sink {sinkName} not found ({sourceList})")

    return moduleIndex, sinkIndex, sourceIndex

def processLog(loggerName, textIoWrapper, level = logging.INFO):
    """
    Logs the process output.

    :param loggerName: the name of the logger.
    :param textIoWrapper: TextIOWrapper with the process output.
    :param level: log level, INFO by default.
    """
    logger = logging.getLogger(loggerName)

    with textIoWrapper:
        for line in textIoWrapper:
            # Lines captured from the recorder have a trailing new line, so it
            # needs to be removed.
            logger.log(level, line.rstrip('\n'))

class Service:
    """
    Class to set up and tear down the needed elements to record a call.

    To record a call a virtual display server and an audio sink are created.
    Then a browser is launched in kiosk mode inside the virtual display server,
    and its audio is routed to the audio sink. This ensures that several
    Services / browsers can be running at the same time without interfering with
    each other, and that the virtual display driver will only show the browser
    contents, without any browser UI. Then the call is joined in the browser,
    and an FFMPEG process to record the virtual display driver and the audio
    sink is started.

    Once the recording is stopped the helper elements are also stopped and the
    recording is uploaded to the Nextcloud server.

    "start()" blocks until the recording ends, so "start()" and "stop()" are
    expected to be called from different threads.
    """

    def __init__(self, backend, token, status, owner):
        self._logger = logging.getLogger(f"{__name__}-{backend}-{token}")

        self.backend = backend
        self.token = token
        self.status = status
        self.owner = owner

        self._started = Event()
        self._stopped = Event()

        self._display = None
        self._audioModuleIndex = None
        self._participant = None
        self._process = None
        self._fileName = None
        self._speakerTimelineFileName = None
        self._speakerTimelineRecordingOffset = 0
        self._speakerTimelineClockUncertainty = 0
        self._participantTracksWriter = None
        self._participantTracksRecordingOffset = 0
        self._participantTracksClockUncertainty = 0
        self._participantTracksDrainStop = Event()
        self._participantTracksDrainThread = None
        self._participantTracksManifestFileName = None

        self._recordingTimeStart = 0
        self._recordingTimeStop = 0

    def __del__(self):
        self._stopHelpers()

    def start(self, actorType, actorId):
        """
        Starts the recording.

        This method blocks until the recording ends.

        :param actorType: the actor type of the Talk participant that started
               the recording.
        :param actorId: the actor id of the Talk participant that started the
               recording.
        :raise Exception: if the recording ends unexpectedly (including if it
               could not be started).
        """

        width = config.getBackendVideoWidth(self.backend)
        height = config.getBackendVideoHeight(self.backend)

        directory = config.getBackendDirectory(self.backend).rstrip('/')

        sanitizedBackend = ''.join([character for character in self.backend if character.isalnum()])

        fullDirectory = f'{directory}/{sanitizedBackend}/{self.token}'

        try:
            # Ensure that PulseAudio is running.
            # A "long" timeout is used to prevent it from exiting before the
            # call was joined.
            subprocess.run(['pulseaudio', '--start', '--exit-idle-time=120'], check=True)

            # Ensure that the directory to start the recordings exists.
            os.makedirs(fullDirectory, exist_ok=True)

            self._display = Display(size=(width, height), manage_global_env=False)
            self._display.start()

            if self._stopped.is_set():
                raise Exception("Display started after recording was stopped")

            # Start new audio sink for the audio output of the browser.
            self._audioModuleIndex, audioSinkIndex, audioSourceIndex = newAudioSink(f"{sanitizedBackend}-{self.token}")
            audioSinkIndex = str(audioSinkIndex)
            audioSourceIndex = str(audioSourceIndex)

            if self._stopped.is_set():
                raise Exception("Audio sink started after recording was stopped")

            env = self._display.env()
            env['PULSE_SINK'] = audioSinkIndex

            browser = config.getBrowserForRecording()

            driverPath = config.getDriverPathForRecording()
            browserPath = config.getBrowserPathForRecording()

            self._logger.debug("Starting participant")
            self._participant = Participant(browser, self.backend, width, height, env, driverPath, browserPath, self._logger)

            self._logger.debug("Joining call")
            self._participant.joinCall(self.token)

            if self._stopped.is_set():
                # Not strictly needed, as if the participant is started or the
                # call is joined after the recording was stopped there will be
                # no display and it will automatically fail, but just in case.
                raise Exception("Call joined after recording was stopped")

            self._started.set()

            BackendNotifier.started(self.backend, self.token, self.status, actorType, actorId)

            extensionlessFileName = f'{fullDirectory}/Recording {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'

            recorderArgumentsBuilder = RecorderArgumentsBuilder()
            recorderArguments = recorderArgumentsBuilder.getRecorderArguments(self.status, self._display.new_display_var, audioSourceIndex, width, height, extensionlessFileName)

            self._fileName = recorderArguments[-1]

            self._logger.debug("Starting recorder")
            self._recordingTimeStart = time.monotonic()
            # pylint: disable=consider-using-with
            self._process = subprocess.Popen(recorderArguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # Log recorder output.
            Thread(target=processLog, args=[f"{__name__}.recorder-{self.backend}-{self.token}", self._process.stdout], daemon=True).start()

            if self._stopped.is_set():
                # Not strictly needed, as if the recorder is started after the
                # recording was stopped there will be no display and it will
                # automatically fail, but just in case.
                raise Exception("Call joined after recording was stopped")

            try:
                timelineStartBefore = time.monotonic()
                self._participant.startSpeakerTimeline()
                timelineStartAfter = time.monotonic()
                self._speakerTimelineRecordingOffset = max(0, ((timelineStartBefore + timelineStartAfter) / 2) - self._recordingTimeStart)
                self._speakerTimelineClockUncertainty = max(0, (timelineStartAfter - timelineStartBefore) / 2)
            except:
                self._logger.exception("Error when starting speaker timeline")

            if config.getParticipantTracksForRecording():
                self._startParticipantTracks()

            returnCode = self._process.wait()

            # recorder process will be explicitly terminated when needed, which
            # returns with 255; any other return code means that it ended
            # without an expected reason.
            if returnCode != 255:
                raise Exception("recorder ended unexpectedly")
        except Exception:
            self._stopHelpers()

            if self._stopped.is_set() and not self._started.is_set():
                # If the service fails before being started but it was already
                # stopped the error does not need to be notified; the error was
                # probably caused by stopping the helpers, and even if it was
                # something else it does not need to be notified either given
                # that the recording was not started yet.
                raise

            try:
                BackendNotifier.failed(self.backend, self.token)
            except:
                pass

            raise

    def stop(self, actorType, actorId):
        """
        Stops the recording and uploads it.

        The recording is removed from the temporary directory once uploaded,
        although it is kept if the upload fails.

        :param actorType: the actor type of the Talk participant that stopped
               the recording.
        :param actorId: the actor id of the Talk participant that stopped the
               recording.
        :raise Exception: if the file could not be uploaded.
        """

        self._stopped.set()

        self._stopParticipantTracks()

        if self._participant and self._fileName:
            try:
                speakerTimeline = self._participant.stopSpeakerTimeline(
                    self._speakerTimelineRecordingOffset,
                    self._speakerTimelineClockUncertainty,
                )
                if speakerTimeline:
                    self._speakerTimelineFileName = self._fileName + '.speakers.json'
                    with open(self._speakerTimelineFileName, 'w', encoding='utf-8') as speakerTimelineFile:
                        json.dump(speakerTimeline, speakerTimelineFile)
            except:
                self._logger.exception("Error when collecting speaker timeline")

        self._stopHelpers()

        BackendNotifier.stopped(self.backend, self.token, actorType, actorId)

        if not self._fileName:
            self._logger.error("Recording stopping before starting, nothing to upload")

            return

        if not os.path.exists(self._fileName):
            self._logger.error("Recording can not be uploaded, %s does not exist", self._fileName)

            return

        participantTracksFileName = self._prepareParticipantTracks()

        BackendNotifier.uploadRecording(self.backend, self.token, self._fileName, self.owner, self._speakerTimelineFileName, participantTracksFileName)

        os.remove(self._fileName)
        if self._speakerTimelineFileName and os.path.exists(self._speakerTimelineFileName):
            os.remove(self._speakerTimelineFileName)
        if self._participantTracksWriter:
            shutil.rmtree(self._participantTracksWriter.directory, ignore_errors=True)

    def _startParticipantTracks(self):
        try:
            self._participantTracksWriter = ParticipantTracksWriter(self._fileName, self._logger)

            trackStartBefore = time.monotonic()
            trackRecording = self._participant.startTrackRecording()
            trackStartAfter = time.monotonic()

            self._participantTracksWriter.setMimeType(trackRecording.get('mimeType'))
            self._participantTracksRecordingOffset = max(0, ((trackStartBefore + trackStartAfter) / 2) - self._recordingTimeStart)
            self._participantTracksClockUncertainty = max(0, (trackStartAfter - trackStartBefore) / 2)
        except:
            self._logger.exception("Error when starting participant tracks")
            self._participantTracksWriter = None

            return

        self._participantTracksDrainThread = Thread(target=self._drainParticipantTracks, daemon=True)
        self._participantTracksDrainThread.start()

    def _drainParticipantTracks(self):
        # The browser keeps the chunks in memory until drained, so they are
        # periodically written to disk during the recording.
        while not self._participantTracksDrainStop.wait(5):
            try:
                self._participantTracksWriter.write(self._participant.drainTrackChunks())
            except:
                self._logger.exception("Error when draining participant tracks")

    def _stopParticipantTracks(self):
        if not self._participantTracksWriter:
            return

        # Selenium must not be used concurrently, so the drain thread needs to
        # be finished before stopping the track recording.
        self._participantTracksDrainStop.set()
        if self._participantTracksDrainThread:
            self._participantTracksDrainThread.join()

        if not self._participant:
            return

        try:
            manifest = self._participant.stopTrackRecording()
            self._participantTracksWriter.write(self._participant.drainTrackChunks())
            if manifest:
                self._participantTracksManifestFileName = self._participantTracksWriter.writeManifest(
                    manifest,
                    self._participantTracksRecordingOffset,
                    self._participantTracksClockUncertainty,
                )
                self._logger.info("Participant tracks written to %s", self._participantTracksWriter.directory)
        except:
            self._logger.exception("Error when collecting participant tracks")

    def _prepareParticipantTracks(self):
        """
        Aligns the participant tracks with the recording and splits them in
        speech chunks.

        The recorder must have finished writing the recording file.

        :return: the archive to upload, or None if there are no tracks or they
                 could not be prepared (the recording is uploaded anyway).
        """

        if not self._participantTracksManifestFileName:
            return None

        directory = self._participantTracksWriter.directory
        try:
            with open(self._participantTracksManifestFileName, encoding='utf-8') as manifestFile:
                manifest = json.load(manifestFile)

            stepStart = time.monotonic()
            Calibration.calibrate(self._fileName, manifest, directory, self._logger)
            self._logger.info("Participant tracks aligned in %.1f seconds", time.monotonic() - stepStart)

            stepStart = time.monotonic()
            TrackChunker.chunk(manifest, directory, self._logger)
            self._logger.info("Participant tracks split in %d chunks in %.1f seconds", len(manifest['chunks']), time.monotonic() - stepStart)

            with open(self._participantTracksManifestFileName, 'w', encoding='utf-8') as manifestFile:
                json.dump(manifest, manifestFile)

            if not manifest['chunks']:
                return None

            return writeArchive(directory, manifest, os.path.join(directory, 'tracks.zip'))
        except:
            self._logger.exception("Error when preparing participant tracks")

            return None

    def _stopHelpers(self):
        self._participantTracksDrainStop.set()
        self._recordingTimeStop = time.monotonic()

        if self._process:
            self._logger.debug("Stopping recorder")
            try:
                self._process.terminate()
                self._process.wait()
            except:
                self._logger.exception("Error when terminating recorder")
            finally:
                self._process = None

        if self._participant:
            self._logger.debug("Disconnecting from signaling server")
            try:
                self._participant.disconnect()
            except:
                self._logger.exception("Error when disconnecting from signaling server")
            finally:
                self._participant = None

        if self._audioModuleIndex:
            self._logger.debug("Unloading audio module")
            try:
                with pulsectl.Pulse(f"audio-module-{self._audioModuleIndex}-unloader") as pacmd:
                    pacmd.module_unload(self._audioModuleIndex)
            except:
                self._logger.exception("Error when unloading audio module")
            finally:
                self._audioModuleIndex = None

        if self._display:
            self._logger.debug("Stopping display")
            try:
                self._display.stop()
            except:
                self._logger.exception("Error when stopping display")
            finally:
                self._display = None

    def getRecordingDuration(self):
        """
        Returns the duration of the recording in seconds.

        The duration is calculated from the start and stop of the service itself
        and not directly from the file of the recording, so there might be a
        small difference with the actual duration of the file.

        :return: The duration of the recording in seconds.
        """

        if self._recordingTimeStart == 0:
            return 0

        if self._recordingTimeStop == 0:
            return round(time.monotonic() - self._recordingTimeStart)

        return round(self._recordingTimeStop - self._recordingTimeStart)
