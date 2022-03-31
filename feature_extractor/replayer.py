import logging
import multiprocessing
import os
import queue
import signal
import threading
import time
import traceback
from abc import abstractmethod
from typing import List, Optional
from absl import flags
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2 import run_configs
from pysc2.env import environment
from pysc2.env.sc2_env import possible_results, AgentInterfaceFormat
from pysc2.env.environment import TimeStep
from pysc2.lib import features, point, replay, point_flag
from pysc2.lib.actions import FunctionCall

__author__ = 'Jesse Hostetler and Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

FLAGS = flags.FLAGS
flags.DEFINE_integer('step_mul', 8, 'Game steps per observation.', allow_override=True)
flags.DEFINE_integer('parallel', 1, 'How many instances to run in parallel.')
flags.DEFINE_string('replay_sc2_version', None,
                    'SC2 version to use for replay. Either "x.y.z" or "latest". If not specified,'
                    ' version is inferred from the replay file. This ought to work, but if that'
                    ' specific version is missing (which seems to happen on Windows), it will'
                    ' raise an error.', allow_override=True)
flags.DEFINE_string('replays', None, 'Path to a directory of replays.')
point_flag.DEFINE_point('window_size', '640,480', 'SC2 window size.')
flags.DEFINE_string('episodes', 'episodes.csv', 'Name of the file containing the episode breaks.')
flags.mark_flag_as_required('replays')


# ----------------------------------------------------------------------------

class DebugStepListener(object):
    """ Processes data from individual replays.
    """

    @abstractmethod
    def start_replay(self, replay_path: str, replay_info: sc_pb.ResponseReplayInfo, player_perspective: int):
        """ Called when starting a new replay.

        Parameters:
          `replay_name`: Path to the replay file.
          `replay_info`: sc2api.ResponseReplayInfo protobuf message
          `player_perspective`: Integer ID of player whose perspective we see
        """
        pass

    @abstractmethod
    def finish_replay(self):
        """ Called when finished with the current replay.
        """
        pass

    @abstractmethod
    def reset(self, pb_obs: sc_pb.ResponseObservation, agent_obs: TimeStep):
        """ Called for the first observation of the replay.

        Parameters:
          `pb_obs`: The observation in protobuf form
          `agent_obs`: The observation in pysc2 features
        """
        pass

    @abstractmethod
    def step(self,
             ep: int,
             step: int,
             pb_obs: sc_pb.ResponseObservation,
             agent_obs: TimeStep,
             agent_actions: List[FunctionCall]):
        """ Called after each agent action step.

        Parameters:
          `ep`: episode number
          `step`: step number
          `pb_obs`: The observation in protobuf form
          `agent_obs`: The observation in pysc2 features
          `agent_actions`: List of actions executed by the agent between the
            previous observation and the current observation.
        """
        pass


class DebugReplayProcessor(object):
    """ Tells the processing code how to configure the game, and instantiates the
    listeners that actually process the data.
    """

    @property
    def interface(self) -> sc_pb.InterfaceOptions:
        """ Returns a sc2api.InterfaceOptions protobuf message.
        """
        size = point.Point(64, 64)
        interface = sc_pb.InterfaceOptions(
            raw=True, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=24))
        size.assign_to(interface.feature_layer.resolution)
        size.assign_to(interface.feature_layer.minimap_resolution)
        return interface

    @property
    def agent_interface_format(self) -> Optional[AgentInterfaceFormat]:
        """
        Defines extra options for the agent's interface. Must match `interface` options where appropriate.
        None means that features are generated according to controller/game info.
        :rtype: AgentInterfaceFormat
        """
        # example that matches interface
        # return features.parse_agent_interface_format(feature_screen=64, feature_minimap=64
        return None

    @property
    def step_mul(self) -> int:
        """ How many game steps per agent step.
        """
        return FLAGS.step_mul

    @property
    def discount(self) -> float:
        """ MDP discount factor.
        """
        return 1.0

    @property
    def score_index(self) -> int:
        """ Index of the score to use for the reward. If `None`, use the outcome of
        the game {-1, 0, 1}.

        TODO: This is supposed to be inferred from the pysc2 `Map` instance, but
        I don't know how to determine `Map` from a replay file.
        """
        return 0

    def valid_replay(self, info, ping) -> bool:
        """ Predicate determining if a replay is valid.

        Parameters:
          `info`: ResponseReplayInfo protobuf message
          `ping`: ResponsePing protobuf message
        """
        return True

    def create_listeners(self) -> List[DebugStepListener]:
        """ Returns a list of Listeners that will process the actual replay data.
        This has to be deferred because the replays are processed in sub-processes.
        This way we can have different Listener instances for each process and we
        don't have to worry about concurrency.
        """
        return [DebugStepListener()]


# ----------------------------------------------------------------------------

class ReplayProcess(multiprocessing.Process):
    """A Process that pulls replays and processes them."""

    def __init__(self, proc_id, run_config, replay_queue, processor, ep_breaks, player_ids=None):
        super(ReplayProcess, self).__init__()
        self.proc_id = proc_id
        self.run_config = run_config
        self.replay_queue = replay_queue
        self.ep_breaks = ep_breaks
        self.player_ids = player_ids

        self.processor = processor
        self._default_score_index = (
            self.processor.score_index if self.processor.score_index is not None
            else -1)
        self._default_score_multiplier = 1  # TODO: Obtain correct value
        self.sinks = self.processor.create_listeners()

    def run(self):
        # Needed to force subprocess to parse flags
        import sys
        FLAGS(sys.argv)

        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Exit quietly.
        replay_name = "none"
        want_rgb = self.processor.interface.HasField("render")
        while True:
            self._print("Starting up a new SC2 instance.")
            try:
                with self.run_config.start(want_rgb=want_rgb, window_size=FLAGS.window_size) as controller:
                    self._print("SC2 Started successfully.")
                    ping = controller.ping()
                    for _ in range(300):
                        try:
                            replay_path = self.replay_queue.get_nowait()
                        except queue.Empty:
                            self._print("Empty queue; returning")
                            return
                        try:
                            replay_name = os.path.basename(replay_path)
                            self._print("Got replay: %s" % replay_path)
                            replay_data = self.run_config.replay_data(replay_path)
                            info = controller.replay_info(replay_data)
                            # self._print((" Replay Info %s " % replay_name).center(60, "-"))
                            # self._print(info)
                            # self._print("-" * 60)
                            if self.processor.valid_replay(info, ping):
                                map_data = None
                                if info.local_map_path:
                                    map_data = self.run_config.map_data(info.local_map_path)
                                for player_info in info.player_info:
                                    player_info = player_info.player_info
                                    self._print(
                                        "Starting %s from player %s's (%s) perspective (%i game loops, %i secs)" % (
                                            replay_name, player_info.player_id, player_info.player_name,
                                            info.game_duration_loops, info.game_duration_seconds))
                                    if self.player_ids is None or \
                                            player_info.player_id == self.player_ids or \
                                            isinstance(self.player_ids, list) and \
                                            player_info.player_id in self.player_ids:
                                        self.process_replay(
                                            controller, replay_path, replay_data, map_data, player_info.player_id)
                            else:
                                self._print("Replay is invalid.")
                        # except Exception as e:
                        #     print(f'Exception type {type(e)}: {e}')
                        finally:
                            self.replay_queue.task_done()
            # except (protocol.ConnectionError, protocol.ProtocolError,
            # remote_controller.RequestError):
            except KeyboardInterrupt:
                return
            # except Exception as ex:
            #     self._print(traceback.format_exc())
            #     return

    def _print(self, s):
        for line in str(s).strip().splitlines():
            logging.info(f'[{self.proc_id}] {line}')

    def process_replay(self, controller, replay_path, replay_data, map_data,
                       player_id):
        """Process a single replay, updating the stats."""
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=self.processor.interface,
            observed_player_id=player_id))
        episode_steps = 0
        last_score = 0
        state = environment.StepType.FIRST

        prev_pb_obs = controller.observe()

        feat = features.features_from_game_info(controller.game_info(), self.processor.agent_interface_format)
        # TODO: Where can we find map_inst?
        score_index = self._default_score_index
        score_multiplier = self._default_score_multiplier

        # get_default(self._default_score_index, map_inst.score_index)

        def zero_on_first_step(value):
            return 0.0 if state == environment.StepType.FIRST else value

        # TODO: Handle multiple agents (if possible in a replay)
        def process_obs(pb_obs):
            nonlocal last_score
            nonlocal state
            nonlocal episode
            nonlocal episode_steps
            prev_score = last_score
            agent_obs = feat.transform_obs(pb_obs)
            # Actions
            agent_actions = []
            for action in pb_obs.actions:
                act_fl = action.action_feature_layer
                try:
                    a = feat.reverse_action(action)
                    agent_actions.append(a)
                except ValueError:
                    try:
                        a = feat.reverse_raw_action(action, agent_obs)
                        agent_actions.append(a)
                    except ValueError:
                        self._print("WARNING: reverse_action() failed:\n%s", action)

            # Done
            outcome = 0
            discount = self.processor.discount
            episode_complete = bool(pb_obs.player_result)

            if episode_complete:
                # self._print( "Episode complete!" )
                state = environment.StepType.LAST
                discount = 0
                player_id = pb_obs.observation.player_common.player_id
                for result in pb_obs.player_result:
                    if result.player_id == player_id:
                        outcome = possible_results.get(result.result, 0)

            # Reward
            if score_index >= 0:  # Game score, not win/loss reward.
                cur_score = agent_obs["score_cumulative"][score_index]
                if episode_steps == 0:  # First reward is always 0.
                    reward = 0
                else:
                    reward = cur_score - last_score
                last_score = cur_score
            else:
                reward = outcome
            # self._print(f'reward: {reward}')

            # checks new episode via special chat message
            if pb_obs.chat:
                if any(chat_msg.message == 'new-episode' for chat_msg in pb_obs.chat):
                    self._print(f'Episode {episode} ended at {total_steps - 1}, score: {prev_score}')
                    state = environment.StepType.FIRST
                    episode_steps = 0
                    episode += 1

            timestep = environment.TimeStep(
                step_type=state,
                reward=zero_on_first_step(reward * score_multiplier),
                discount=zero_on_first_step(discount),
                observation=agent_obs)

            return (pb_obs, timestep, agent_actions)

        def step():
            nonlocal state

            controller.step(self.processor.step_mul)
            if state == environment.StepType.FIRST:
                state = environment.StepType.MID
            elif state == environment.StepType.LAST:
                state = environment.StepType.FIRST
            # Change to LAST handled in process_obs()

            # Observations
            pb_obs = controller.observe()
            pb_obs, agent_obs, agent_actions = process_obs(pb_obs)
            prev_pb_obs = pb_obs
            return (pb_obs, agent_obs, agent_actions)

        # Step through the replay
        replay_info = controller.replay_info(replay_data)

        for s in self.sinks:
            s.start_replay(replay_path, replay_info, player_id)

        pb_obs, agent_obs, agent_actions = process_obs(prev_pb_obs)
        for s in self.sinks:
            s.reset(pb_obs, agent_obs)

        episode = 0
        episode_steps = 0
        total_steps = 0

        while True:
            pb_obs, agent_obs, agent_actions = step()

            for s in self.sinks:
                s.step(episode, episode_steps, pb_obs, agent_obs, agent_actions)

            if pb_obs.player_result:
                break

            if agent_obs.step_type == environment.StepType.LAST or total_steps in self.ep_breaks:
                self._print(f'Episode {episode} ended at {total_steps}, score: {last_score}')
                episode_steps = 0
                episode += 1
                total_steps += 1
                continue

            total_steps += 1
            episode_steps += 1

        self._print(f'Episode {episode} ended at {total_steps}, score: {last_score}')
        for s in self.sinks:
            s.finish_replay()


def replay_queue_filler(replay_queue, replay_list):
    """A thread that fills the replay_queue with replay filenames."""
    for replay_path in replay_list:
        replay_queue.put(replay_path)
    logging.info("Done filling queue")


def replay_paths(replay_dir):
    """A generator yielding the full path to the replays under `replay_dir`."""
    replay_dir = os.path.abspath(replay_dir)
    if replay_dir.lower().endswith(".sc2replay"):
        yield replay_dir
        return
    for f in os.listdir(replay_dir):
        if f.lower().endswith(".sc2replay"):
            yield os.path.join(replay_dir, f)


class ReplayProcessRunner(object):
    """ Driver class for replay processing.
    """

    def __init__(self, replay_dir, replay_processor, sc2_version=None, parallel=1,
                 ep_break_file='episodes.csv', player_ids=None, amount=None):
        """
        Parameters:
          `replay_dir`: Directory containing replay files to process
          `replay_processor`: ReplayProcessor instance
          `sc2_version`: `str`: `None`, 'x.y.z', or 'latest'. If `None`, version
            is inferred from replay file.
          `parallel`: How many parallel processes to run.
        """
        self.replay_dir = replay_dir
        self.replay_processor = replay_processor
        self.sc2_version = sc2_version
        self.parallel = parallel
        self.player_ids = player_ids
        self.amount = amount

        # checks for episode breaks
        self.ep_breaks = []
        ep_break_file = os.path.join(replay_dir, ep_break_file)
        if os.path.isfile(ep_break_file):
            with open(ep_break_file, 'r') as file:
                self.ep_breaks = set([int(ep) for ep in file.read().split(',')])

    def run(self):
        """ Process all of the replay files.
        """
        run_config = run_configs.get()

        if not os.path.exists(self.replay_dir):
            raise RuntimeError(f'Specified replay dir={self.replay_dir} doesn\'t exist.')

        try:
            # For some reason buffering everything into a JoinableQueue makes the
            # program not exit, so save it into a list then slowly fill it into the
            # queue in a separate thread. Grab the list synchronously so we know there
            # is work in the queue before the SC2 processes actually run, otherwise
            # The replay_queue.join below succeeds without doing any work, and exits.
            logging.info(f'Getting replay list: {self.replay_dir}')
            replay_list = sorted(replay_paths(self.replay_dir))
            logging.info(f'{len(replay_list)} replays found.')
            if self.amount is not None and len(replay_list) > self.amount:
                replay_list = replay_list[:self.amount]
                logging.info(f'Selected the first {self.amount} replays.')

            if len(self.ep_breaks) > 0:
                logging.info(f'Using {len(self.ep_breaks)} episode breaks.')
            if not replay_list:
                return

            if not self.sc2_version:
                version = replay.get_replay_version(
                    run_config.replay_data(replay_list[0]))
                run_config = run_configs.get(version=version)
                # FIXME: Validate that needed version exists
                # run_config = run_configs.get(version="latest")
                logging.info(f'Assuming version: {version.game_version}')
            else:
                run_config = run_configs.get(version=self.sc2_version)

            logging.info('')

            replay_queue = multiprocessing.JoinableQueue(self.parallel * 10)
            replay_queue_thread = threading.Thread(target=replay_queue_filler,
                                                   args=(replay_queue, replay_list))
            replay_queue_thread.start()
            time.sleep(1)

            logging.info('Started replay_queue_thread')

            procs = []
            for i in range(min(len(replay_list), self.parallel)):
                p = ReplayProcess(i, run_config, replay_queue, self.replay_processor, self.ep_breaks, self.player_ids)
                procs.append(p)
                logging.info(f'Starting process {i}')
                p.start()
                # Stagger startups, otherwise they seem to conflict somehow
                time.sleep(1)

            replay_queue.join()  # Wait for the queue to empty.
        except KeyboardInterrupt:
            logging.info('Caught KeyboardInterrupt, exiting.')
