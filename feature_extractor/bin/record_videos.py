import logging
import os
import time
import skvideo
import numpy as np
import multiprocessing as mp
import threading as td
from typing import Tuple
from absl import flags, app
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.env.sc2_env import AgentInterfaceFormat, Dimensions
from feature_extractor.replayer import DebugReplayProcessor, DebugStepListener, ReplayProcessRunner
from feature_extractor.util.io import create_clear_dir, get_file_name_without_extension, save_dict_json, \
    get_files_with_extension
from feature_extractor.util.logging import change_log_handler
from feature_extractor.util.screen import get_window, get_window_image

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Loads replays and saves a video file of the main game screen for each replay and episode therein.'

FLAGS = flags.FLAGS
flags.DEFINE_string('output', 'output', 'Path to the directory in which to save the video files.')
flags.DEFINE_integer('amount', None, 'Number of videos to be recorded given the input set. '
                                     'If not specified, it results in recording all replays.')
flags.DEFINE_float('fps', 10, 'The frames per second ratio used to save the videos.')
flags.DEFINE_integer('crf', 18, 'Video constant rate factor: the default quality setting in `[0, 51]`')
flags.DEFINE_bool('hide_hud', True, 'Whether to hide the HUD / information panel at the bottom of the screen.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')
flags.DEFINE_bool('resume', False, 'Whether to resume a previous recording session. If True, `clear` will be ignored.')

CAPTURE_CMD = 'capture'
START_CMD = 'start'

SC2_WINDOW_NAME = 'StarCraft II'
SC2_WINDOW_OWNER = 'SC2'
PLAYER_ID = 1
BTW_REPLAYS_SLEEP = 1


class _ReplayProcessor(DebugReplayProcessor):

    def __init__(self, output: str, queue: mp.Queue, step_mul: int = 1,
                 window_size: Tuple[int, int] = (640, 480),
                 player_id: int = 1, hide_hud: bool = False):
        """
        Creates a new video recorder processor.
        :param str output: the path in which to save the replay video recordings.
        :param mp.Queue queue: the queue to send commands to a video recording thread.
        :param int step_mul: the environment's step multiplier.
        :param (int, int) window_size: SC2 window size.
        :param int player_id: the id of the player considered as the agent in the replay file.
        :param bool hide_hud: whether to hide the HUD / information panel at the bottom of the screen.
        """
        self._step_mul = step_mul

        self._aif = AgentInterfaceFormat(
            rgb_dimensions=Dimensions((window_size[0], window_size[1]), 1) if hide_hud else None)

        self._interface_options = sc_pb.InterfaceOptions(
            raw=True, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=self._aif.camera_width_world_units))
        if self._aif.feature_dimensions is not None:
            self._aif.feature_dimensions.screen.assign_to(self._interface_options.feature_layer.resolution)
            self._aif.feature_dimensions.minimap.assign_to(self._interface_options.feature_layer.minimap_resolution)
        if self._aif.rgb_dimensions is not None:
            self._aif.rgb_dimensions.screen.assign_to(self._interface_options.render.resolution)
            self._aif.rgb_dimensions.minimap.assign_to(self._interface_options.render.minimap_resolution)

        self._listener = _StepListener(output, queue, player_id)

    @property
    def step_mul(self):
        return self._step_mul

    @property
    def interface(self):
        return self._interface_options

    @property
    def agent_interface_format(self):
        return self._aif

    def create_listeners(self):
        return [self._listener]


class _StepListener(DebugStepListener):

    def __init__(self, output: str, queue: mp.Queue, player_id: int = 1):
        self.output = output
        self.queue = queue
        self.player_id = player_id

        self._ignore_replay: bool = False
        self._total_eps: int = -1
        self._total_steps: int = 0
        self._replay_name: str = ''

    def start_replay(self, replay_name, replay_info, player_perspective):
        # checks perspective, ignore if not player's side
        self._ignore_replay = player_perspective != self.player_id

        if not self._ignore_replay:
            self._replay_name = os.path.basename(replay_name)
            logging.info(f'Starting replay \'{replay_name}\'...')

    def _start_new_episode(self):

        # send start new video command
        self.queue.put('start')

        # send output file name to recorder to start a new video
        file_name = get_file_name_without_extension(self._replay_name)
        output_file = os.path.join(self.output, f'{file_name}.mp4')
        if os.path.isfile(output_file):
            # add suffix in case file already exists (multiple episodes per replay)
            output_file = os.path.join(self.output, f'{file_name}-{self._total_eps}.mp4')

        # wait for ack, send output video file name and wait for new ack
        self.queue.join()
        self.queue.put(output_file)
        self.queue.join()

        logging.info(f'Starting episode {self._total_eps} of replay \'{self._replay_name}\'...')

    def reset(self, pb_obs, agent_obs):
        self._total_eps = -1
        self._total_steps = 0

    def step(self, ep, step, pb_obs, agent_obs, agent_actions):
        # check ignore
        if self._ignore_replay:
            return

        # checks new episode
        if ep != self._total_eps:
            self._total_eps = ep
            self._start_new_episode()

        # send capture screen command and wait for ack
        self.queue.put(CAPTURE_CMD)
        self.queue.join()

    def finish_replay(self):
        if self._ignore_replay:
            return

        logging.info(
            f'Collected {self._total_steps} frames from {self._total_eps} episodes '
            f'from replay \'{self._replay_name}\'...')


class _VideoRecorderThread(td.Thread):

    def __init__(self, cmd_queue: mp.Queue, resolution: Tuple[int, int] = (640, 480),
                 fps: float = 22.5, crf: int = 18, idx=0):
        """
        Creates a new thread to receive commands to capture window frames.
        :param mp.Queue cmd_queue: the queue from which to receive commands from controlling processes.
        :param (int,int) resolution: the target resolution of the SC2 window.
        :param float fps: the frames-per-second at which videos are to be recorded.
        :param int crf: constant rate factor (CRF): the default quality (and rate control) setting in `[0, 51]`, where
        lower values would result in better quality, at the expense of higher file sizes.
        :param int idx: the index of the SC2 producer process, used to identify the SC2 screen window if multiple are
        active.
        """
        super().__init__()
        self.queue = cmd_queue
        self.resolution = resolution
        self.fps = fps
        self.crf = crf
        self.idx = idx

        self._sc2_window_id: int = -1
        self._video_writer: skvideo.io.FFmpegWriter = None

    def run(self):

        # wait for commands from a controlling process
        while True:

            cmd = self.queue.get()
            if cmd is None:
                # checks writer
                if self._video_writer is not None:
                    self._video_writer.close()
                self._video_writer = None

                logging.info('[VideoRecorder] Done, exiting.')
                return

            if cmd == START_CMD:
                # checks writer
                if self._video_writer is not None:
                    self._video_writer.close()
                self._video_writer = None

                # reset and try to get window
                self._sc2_window_id = -1
                wid = get_window(SC2_WINDOW_NAME, exact_match=True, owner=SC2_WINDOW_OWNER)
                if wid is None:
                    logging.info('[VideoRecorder] Could not find StarCraftII window!')
                    self._sc2_window_id = -1
                else:
                    self._sc2_window_id = wid
                    logging.info(f'[VideoRecorder] Found SC2 window: {self._sc2_window_id}')

                # gets video file name
                self.queue.task_done()
                output_file = self.queue.get()

                # creates new video recorder
                self._video_writer = skvideo.io.FFmpegWriter(
                    output_file,
                    inputdict={'-r': str(self.fps)},
                    outputdict={'-crf': str(self.crf), '-pix_fmt': 'yuv420p'})
                logging.info(f'[VideoRecorder] Recording new video to: {output_file}...')

            elif cmd == CAPTURE_CMD and self._sc2_window_id != -1:
                # captures new frame
                img = get_window_image(self._sc2_window_id)
                if img is not None:
                    img = np.asarray(img)
                    # check for correct resolution, if not try to correct it (this problem occurs on Windows)
                    if (img.shape[1], img.shape[0]) != self.resolution:
                        v_diff = max(0, img.shape[0] - self.resolution[1])
                        h_diff = max(0, img.shape[1] - self.resolution[0])
                        img = img[v_diff:, h_diff:, :]
                    self._video_writer.writeFrame(img)

            else:
                logging.info(f'[VideoRecorder] Invalid command received: {cmd}!')

            # send a confirmation to resume processing on the other side
            self.queue.task_done()


def main(unused_args):
    args = flags.FLAGS

    # checks output dir and log file, save args
    out_dir = args.output
    create_clear_dir(out_dir, not args.resume and args.clear)
    change_log_handler(os.path.join(out_dir, 'record_video.log'), args.verbosity)
    save_dict_json({a: args[a].value if hasattr(args[a], 'value') else str(args[a]) for a in args},
                   os.path.join(args.output, 'args.json'))

    logging.info('===================================================================')

    # check input
    if not os.path.isdir(args.replays):
        raise ValueError(f'Replay directory does not exist: {args.replays}!')
    replay_files = get_files_with_extension(args.replays, 'SC2Replay', sort=True)

    # select from replays if needed
    if args.amount is not None and len(replay_files) > args.amount:
        replay_files = replay_files[:args.amount]
        logging.info(f'Selected the first {args.amount} replays from {args.replays}.')
    else:
        logging.info(f'Selected all {len(replay_files)} replays from {args.replays}.')
    if args.resume:
        logging.info('Resuming recording task')

    # replays and records each file, one by one
    for i, replay_file in enumerate(replay_files):
        logging.info('===================================================================')

        if args.resume:
            # skip if existing video
            file_name = get_file_name_without_extension(replay_file)
            output_file = os.path.join(out_dir, f'{file_name}.mp4')
            if os.path.isfile(output_file):
                logging.info(f'Video(s) already captured for replay {replay_file}, skipping...')
                continue

        logging.info(f'Processing replay {i}/{len(replay_files)}...')

        # creates and starts video recording thread
        sync_queue = mp.JoinableQueue()
        consumer_thread = _VideoRecorderThread(sync_queue, tuple(args.window_size), args.fps, args.crf)
        consumer_thread.start()

        # creates and runs the replay processor
        sample_processor = _ReplayProcessor(
            out_dir, sync_queue, args.step_mul, tuple(args.window_size), PLAYER_ID, hide_hud=True)

        replayer_processor = ReplayProcessRunner(
            replay_file, sample_processor, args.replay_sc2_version, args.parallel, player_ids=PLAYER_ID)
        replayer_processor.run()

        # terminate thread
        sync_queue.put(None)
        consumer_thread.join()
        logging.info(f'Sleeping {BTW_REPLAYS_SLEEP}s...')
        time.sleep(BTW_REPLAYS_SLEEP)

    logging.info('===================================================================')
    logging.info('Done!')


if __name__ == "__main__":
    app.run(main)
