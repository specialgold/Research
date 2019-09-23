import RCPSP_fileparser as fp

class Parameters:
    def __init__(self, info):

        self.simu_len = 10000

        self.jobs = info.jobs
        self.act_job = info.jobs-2
        # self.horizon = info.horizon
        self.horizon = 86
        self.rep_horizon = 20
        self.renewable = info.renewable
        self.nonrenewable = info.nonrenewable
        self.doubly_constrained = info.doubly_constrained

        self.releaseDate = info.releaseDate

        self.dueDate = info.dueDate

        self.modes = info.modes
        self.successors = info.successors
        self.requests = info.requests

        self.resourceAvailabilities = info.resourceAvailabilities
        self.num_queue = 5
        self.maxRes_Avbl = 10
        self.maxRes_Req = 10
        self.max_duration = 10
        # self.maxRes_Avbl = info.maxRes_Avbl
        # self.maxRes_Req = info.maxRes_Req

        # self.max_queue_width = self.maxRes_Req
        self.network_input_height = self.rep_horizon # 1 means precedence
        self.network_input_width = (self.maxRes_Avbl + (self.num_queue * self.maxRes_Req)) * self.renewable + 1 # 1 mean backlog
        # self.network_input_width = (self.maxRes_Avbl + (self.act_job * self.max_queue_width)) * (self.renewable + self.nonrenewable)
        # self.network_input_width = ((self.maxRes_Avbl + self.max_queue_width + 1) * self.renewable)  # 1 mean mark of precedence
        self.network_output_height = self.num_queue + 1



        self.delay_penalty = -10 # delay days * penalty
        self.doubly_selected_penalty = -100
        self.over_timehorizon = -100
        self.voidwithoutanything_penalty = -100

        self.allocate_reward = 10
        self.sequence_reward = 2

        self.discount = 1


if __name__ == '__main__':
    info = fp.parser('j102_2.mm')
    pa = Parameters(info)
