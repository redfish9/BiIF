import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
from os import makedirs

from qmix_learner import QLearner
from qmix_parallel_runner import ParallelRunner
from qmix_parallel_runner_mpe import ParallelRunner as ParallelRunner_mpe
from dcntrl_controller import BasicMAC
from components.episode_buffer import ReplayBuffer
from utils.mappo_utils.separated_buffer import SeparatedReplayBuffer
from components.transforms import OneHot

import highway_env
import gym
from gym.envs.registration import register
from envs.env_wrappers import SubprocVecEnv
from envs.env_wrappers_mpe import SubprocVecEnv as SubprocVecEnv_mpe
from envs.mpe.MPE_env import MPEEnv
import numpy as np


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # Create the local results directory
    if args.local_results_path == "":
        args.local_results_path = dirname(dirname(abspath(__file__)))
    makedirs(args.local_results_path, exist_ok=True)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    date_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    # f"diff={args.env_args['difficulty']}"
    envargs_list = []  # env args we want to appear in name

    algargs_list = [] # [f"act={args.action_selector}"] # alg args we want to appear in name

    namelist = [args.name, args.env, args.label, *envargs_list, *algargs_list, f"seed={args.seed}", date_time]
    namelist = [name.replace("_", "-") for name in namelist if name is not None]
    args.unique_token = "_".join(namelist)

    if args.use_tensorboard:
        tb_logs_direc = os.path.join(args.local_results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(args.unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def get_model_path(checkpoint_path, load_step):
    timesteps = []
    timestep_to_load = 0
    # Go through all files in args.checkpoint_path
    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    if load_step == 0:
        # choose the max timestep
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - load_step))
    model_path = os.path.join(checkpoint_path, str(timestep_to_load))
    print("MODEL PATH IS ", model_path)
    return model_path

def run_sequential(args, logger):
    # Init runner so we can get env info
    print("ENV IS : ", args.env)
    # Init runner so we can get env info
    if args.env == "highway":
        env = make_train_env(args)
    else:
        env = make_train_env_mpe(args)

    if args.env == "highway":
        runner = ParallelRunner(args=args, env=env, logger=logger)
    else:
        runner = ParallelRunner_mpe(args=args, env=env, logger=logger)

    # print("SETUP RUNNER")
    # Set up schemes and groups here
    env_info = runner.get_env_info(args)
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    # Initialize the folder to store the screenshots
    if args.animation_enable:
        if not os.path.isdir("animation"):
            os.makedirs("animation")
        for i in range(args.batch_size_run):
            if not os.path.isdir("animation/" + str(i)):
                os.makedirs("animation/" + str(i))

    print("EPISODE LIMIT IS ", env_info["episode_limit"])
    # sys.exit(0)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "speed": {"vshape": (1,), "group": "agents"},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "agent_terminated": {"vshape": (1,), "group": "agents", "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)


    # Setup multiagent controller here
    mac = BasicMAC(buffer.scheme, groups, args)
    # print("SETUP MAC")

    # Learner
    learner = QLearner(mac, buffer.scheme, logger, args)
    # print("SETUP LEARNER")

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    # print("SETUP RUNNER")

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch, avg_win_rates, avg_rwd, avg_len = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        # n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env

        # save models
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

        log_print(args, episode, runner.t_env, avg_win_rates, avg_rwd, avg_len)

        # Enable the navigation metrics printing for highway_env
        if args.metrics_enable and args.env == "highway":
            print("Navigation Metrics Print")
            print("--------------------------------")
            for i in range(int(args.num_test_episodes // args.batch_size_run)):
                # Run for a whole episode at a time
                avg_speed, avg_survival_time, win_num = metric_generate(episode_batch)
                for j in range(args.batch_size_run):
                    episode_test = args.batch_size_run * i + j
                    metric_log_print(args, episode_test, avg_speed[j], avg_survival_time[j], win_num[j])

        episode += args.batch_size_run

    runner.close_env()
    logger.console_logger.info("Finished Training")

# Helper function to generate the metrics of Highway
def metric_generate(episode_batch):
    speed_raw = episode_batch["speed"][:, :-1]
    n_threads, episode_len, n_agents, _ = speed_raw.shape
    speed_raw = speed_raw.reshape(n_threads, episode_len, n_agents)

    terminate_mask_raw = 1 - episode_batch["agent_terminated"][:, :-1]
    terminate_mask_raw = terminate_mask_raw.reshape(n_threads, episode_len, n_agents)

    avg_speed = speed_raw * terminate_mask_raw

    avg_survival_time = terminate_mask_raw.sum(axis=1)

    win_num = np.zeros((n_threads,))
    for i in range(n_threads):
        win_num[i] = sum([avg_survival_time[i, j] == episode_len for j in range(n_agents)])

    avg_speed = avg_speed.sum(axis=1) / (avg_survival_time + 1E-10)
    avg_speed = np.average(np.array(avg_speed.cpu().numpy()), axis=1)

    avg_survival_time = np.average(np.array(avg_survival_time.cpu().numpy()), axis=1)
    return avg_speed, avg_survival_time, win_num


def log_print(args, episode, t_env, episode_win_rate, episode_reward, episode_len, behavior_loss=None,
              stability_loss=None, total_loss=None, prediction_loss=None):
    # log printer for highway_env
    if args.env == "highway":
        print("Episode #", episode,
              " | Current time step: ", t_env,
              " | Average Episode Win Num: ", episode_win_rate,
              " | Average Episode Reward: ", episode_reward,
              " | Average Episode Length: ", episode_len)
        print("--------------------------------")
    # log printer for MPE
    elif args.env == "MPE":
        print("Episode #", episode,
              " | Current time step: ", t_env,
              " | Average Episode Reward: ", episode_reward,
              " | Average Episode Length: ", episode_len)
        print("--------------------------------")

def metric_log_print(args, episode, avg_speed, avg_survival_time, win_num):
    print("Episode #", episode,
          " | Average Episode Win Num: ", win_num,
          " | Average Episode Survival Time: ", avg_survival_time,
          " | Average Episode Speed: ", avg_speed)
    print("--------------------------------")

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    return config


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = env_wrapper(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.batch_size_run == 1:
        return SubprocVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.batch_size_run)])

def make_train_env_mpe(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.batch_size_run == 1:
        return SubprocVecEnv_mpe([get_env_fn(0)])
    else:
        return SubprocVecEnv_mpe([get_env_fn(i) for i in range(all_args.batch_size_run)])


# env wrapper for highway_env
# env wrapper for highway_env
def register_highway_envs():
    register(
        id='intention-v0',
        entry_point='highway_env.envs:IntentionEnv'
    )

    register(
        id='intention-v1',
        entry_point='highway_env.envs:IntentionEnvHard',
    )

# env wrapper for highway_env
def env_wrapper(args):
    register_highway_envs()
    if args.difficulty == "easy":
        env = gym.make("intention-v0")
    else:
        env = gym.make("intention-v1")

    env.configure({
      "action": {
        "type": "MultiAgentAction",
        "action_config": {
          "type": "DiscreteMetaAction",
        }
      }
    })

    env.configure({
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": args.n_obs_vehicles,  # Number of observed vehicles
                "features": ["id", "presence", "x", "y", "vx", "vy",
                             "ax", "ay", "type"],  # added features
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted"
            }
        }
    })
    # State definition
    env.configure({"features": ["id", "presence", "x", "y", "vx", "vy", 
                                "ax", "ay", "type"]})  # added features

    env.configure({"duration": args.episode_limit})
    env.configure({"lanes_count": args.n_lane})
    env.configure({"controlled_vehicles": args.n_agents})
    env.configure({"simulation_frequency": 5})
    # Display
    env.configure({'scaling': args.scaling, 'screen_height': args.screen_height, 'screen_width': args.screen_width})
    env.configure({"vehicles_count": args.n_other_vehicles})
    env = highway_env.envs.MultiAgentWrapper(env)
    return env