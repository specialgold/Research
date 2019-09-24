import numpy as np
import math
import os
import numpy as np

from basic import RCPSP_fileparser as fp
import random
class Parameters:
    def __init__(self):
        self.path = 'data/30/'
        self.seq = 0

        self.dirList = os.listdir(self.path)
        random.shuffle(self.dirList)
        self.total = len(self.dirList)
        self.train = self.dirList[:int(self.total*0.8)]
        self.train_len = len(self.train)
        self.test = self.dirList[int(self.total*0.8):]
        a = np.random.choice(self.train, 1)
        self.file = a[0]
        self.path = self.path + self.file
        self.s_type = 'Max'

        info = fp.parser(self.path)
        # print(a[0])
        self.opt = fp.opt_parser('data/opt30/j30opt.sm')

        self.jobs = info.jobs
        self.act_job = info.jobs - 2
        self.horizon = info.horizon
        self.rep_horizon = 150
        self.renewable = info.renewable
        self.nonrenewable = info.nonrenewable
        self.doubly_constrained = info.doubly_constrained
        self.releaseDate = info.releaseDate

        self.dueDate = info.dueDate
        self.resourceAvailabilities = info.resourceAvailabilities

        self.num_queue = 5
        self.maxRes_Avbl = 10
        self.maxRes_Req = 10
        self.max_duration = 10

        self.modes = info.modes
        self.successors = info.successors
        self.requests = info.requests



        self.allocate_reward = 10
        self.sequence_reward = 2

        ###############################################

        self.output_filename = 'data/tmp'

        self.num_epochs = 5001         # number of training epochs
        self.simu_len = 10             # length of the busy cycle that repeats itself
        self.num_ex = 1                # number of sequences

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline
        self.episode_max_length = 300  # enforcing an artificial terminal

        # self.num_res = 2               # number of resources in the system
        # self.num_nw = self.num_queue                # maximum allowed number of work in the queue

        self.time_horizon = 20         # number of time steps in the graph
        self.max_job_len = 15          # maximum duration of new jobs
        self.res_slot = 10             # maximum number of available resource slots
        self.max_job_size = 10         # maximum resource request of new work

        self.backlog_size = 30         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40          # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor
        # self.discount = 0.9         # discount factor
        # distribution for new job arrival
        # self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation
        # assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        # self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.rep_horizon + 1# 1 means totalduration
        self.network_input_width = (self.maxRes_Avbl + (
                    self.num_queue * self.maxRes_Req)) * self.renewable + 1  # 1 mean backlog
        self.network_output_dim = self.num_queue + 1
        # self.network_input_height = self.time_horizon
        # self.network_input_width = \
        #     (self.res_slot +
        #      self.max_job_size * self.num_nw) * self.num_res + \
        #     self.backlog_width + \
        #     1  # for extra info, 1) time since last new job

        # compact representation
        # self.network_compact_dim = (self.num_res + 1) * \
        #     (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        # self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.network_compact_dim = self.rep_horizon * (self.renewable) \
                                   + self.num_queue * (self.renewable + 2) \
                                   + self.backlog_size
        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"

    def reset(self, type):
        self.path = 'data/30/'
        # self.path = '../data/RCPSP/train/'


        if type == 'train':
            # a = np.random.choice(self.train, 1)
            # file = a[0]
            self.file = self.train[self.seq % self.train_len]
            self.seq += 1
        else:
            if len(self.test) == 0:
                # self.test = self.dirList[:]
                self.test = self.dirList[int(self.total * 0.8):]
                return False
            self.file = self.test[0]
            del self.test[0]

        self.path = 'data/30/' + self.file
        info = fp.parser(self.path)
        # print(a[0])
        self.dueDate = info.dueDate
        # print(self.path)
        # print(self.dueDate)
        self.jobs = info.jobs
        self.act_job = info.jobs - 2
        self.horizon = info.horizon

        self.renewable = info.renewable
        self.nonrenewable = info.nonrenewable
        self.doubly_constrained = info.doubly_constrained
        self.releaseDate = info.releaseDate

        self.dueDate = info.dueDate
        self.resourceAvailabilities = info.resourceAvailabilities

        self.modes = info.modes
        self.successors = info.successors
        self.requests = info.requests
        return True
    # def compute_dependent_parameters(self):
    #     assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
    #     self.backlog_width = self.backlog_size / self.time_horizon
    #     self.network_input_height = self.time_horizon
    #     self.network_input_width = \
    #         (self.res_slot +
    #          self.max_job_size * self.num_nw) * self.num_res + \
    #         self.backlog_width + \
    #         1  # for extra info, 1) time since last new job
    #
    #     # compact representation
    #     self.network_compact_dim = (self.num_res + 1) * \
    #         (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator
    #
    #     self.network_output_dim = self.num_nw + 1  # + 1 for void action

