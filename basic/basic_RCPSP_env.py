import numpy as np
import RCPSP_fileparser as fp
# import basic_param as pa
import parameters_RCPSP as pa
import theano
class Env:
    def __init__(self, pa, repre='compact'):
        self.test_type = 'train'
        self.pa = pa
        self.repre = repre  # image or compact representation
        self.renew = RenewableResources(pa)
        self.nonrenew = NonRenewableResources(pa)
        self.RM = ResourceManagement(self.renew, self.nonrenew, pa)
        self.job_record = JobRecord()
        self.activities = self.getActivities(pa, self.job_record)
        self.backlog = self.activities[1:-1]
        # print(self.backlog)
        self.stage = 0
        self.done_reward=0

        # self.dueDate = pa.dueDate
        # self.releaseDate = pa.releaseDate
        # self.NotDonePenalty = False
        # self.rep_current_time = np.ones((self.pa.horizon, 1))
        self.curr_time = 0


        self.eligible = [None] * self.pa.num_queue
        self.job_slot = JobSlot(self.eligible)
        self.eligible_cnt = 0
        self.allocated = []

        self.backlog_cnt = self.pa.act_job

        self.act_seq = []
        self.s_type = pa.s_type
        self.findEligible(s_type=self.s_type)
        # print(self.backlog)
        # print(self.eligible)

    def reset(self):
        # print("reset")
        result = self.pa.reset(self.test_type)
        if not result:
            return result
        self.job_record = JobRecord()
        self.eligible_cnt = 0
        self.stage = 0
        self.backlog_cnt = self.pa.act_job
        self.curr_time = 0
        self.allocated = []
        self.done_reward = 0
        self.act_seq = []
        self.renew = RenewableResources(self.pa)
        self.nonrenew = NonRenewableResources(self.pa)
        self.RM = ResourceManagement(self.renew, self.nonrenew, self.pa)
        self.eligible = [None] * self.pa.num_queue
        self.job_slot = JobSlot(self.eligible)
        self.activities = self.getActivities(self.pa, self.job_record)
        self.backlog = self.activities[1:-1]
        # self.dueDate = pa.dueDate
        # self.releaseDate = pa.releaseDate
        self.findEligible(s_type=self.s_type)
        return True

    def findEligible(self, s_type='Max'):
        if s_type == 'Max':
            self.findMaxEligible()
        elif s_type == 'Min':
            self.findMinEligible()
        elif s_type == 'Random':
            self.findRandomEligible()
        else:
            print("wrong s_type")

    def findMinEligible(self):
        for j in range(self.pa.num_queue):
            if self.eligible[j] is None:
                tmp = None
                min = 1000
                for act in self.backlog:
                    if np.sum(act.rep_precedence) == 0:
                        if act.after_duration < min:
                            # print(tmp)
                            tmp = act
                            min = act.after_duration
                if tmp is not None:
                    self.eligible[j] = tmp
                    self.eligible_cnt += 1
                    self.backlog.remove(tmp)
                    break


    def findMaxEligible(self):
        for j in range(self.pa.num_queue):
            if self.eligible[j] is None:
                tmp = None
                max = 0
                for act in self.backlog:
                    if np.sum(act.rep_precedence) == 0:
                        if act.after_duration > max:
                            # print(tmp)
                            tmp = act
                            max = act.after_duration
                if tmp is not None:
                    self.eligible[j] = tmp
                    self.eligible_cnt += 1
                    self.backlog.remove(tmp)
                    break

    def findRandomEligible(self):
        for j in range(self.pa.num_queue):
            if self.eligible[j] is None:

                list = []
                for act in self.backlog:
                    if np.sum(act.rep_precedence) == 0:
                        list.append(act)

                if len(list) != 0:
                    tmp = np.random.choice(list, 1)
                    self.eligible[j] = tmp[0]
                    self.eligible_cnt += 1
                    self.backlog.remove(tmp[0])
                    break

    def getActivities(self, pa, job_record):
        activities = []
        for i in range(pa.jobs):
            act = Activity(pa, i)
            activities.append(act)
            if i > 0 and i < pa.jobs -1:
                job_record.record[i-1] = act

        return activities
    def random_action(self):
        # if reward is penalty = true
        return int(np.random.choice(self.pa.network_output_height, 1))


    def observe(self):
        if self.repre == 'image':
            input = np.zeros((self.pa.network_input_height, self.pa.network_input_width))
            # print(input.shape)
            ir_pt = 0

            for i in range(self.pa.renewable):

                input[1:, ir_pt: ir_pt + self.renew.res_slot[i]] = self.renew.forinput[i][:,:]
                ir_pt += self.pa.maxRes_Avbl

                for act in self.eligible:
                    if act is not None:
                        input[0:1, ir_pt: ir_pt + 1] = act.after_duration
                        input[1: 1 + act.durations[0], ir_pt: ir_pt + act.re_requests[0][i]] = 1


                    ir_pt += self.pa.maxRes_Req
            # need to add backlog
            i = 0
            for act in self.backlog:
                input[i:i+1, ir_pt: ir_pt+1] = act.after_duration
                i += 1
            # input[:self.backlog_cnt, ir_pt: ir_pt+1] = 1

            ir_pt += 1
            #
            # print()
            # for i in range(self.pa.network_input_height):
            #     for j in range(int(self.pa.network_input_width)):
            #         print str(input[i, j]) + ' ',
            #         # pass
            #     print()

            return input
        elif self.repre == 'compact':
            compact_repr = np.zeros(self.pa.network_compact_dim,  # backlog indicator
                                    dtype=theano.config.floatX)
            # print(self.pa.time_horizon)
            cr_pt = 0
            # new_avbl_res = self.renew.avbl_slot[t: t + activity.durations[0], :] - activity.re_requests[0]
            for i in range(self.pa.renewable):
                compact_repr[cr_pt:cr_pt + self.pa.rep_horizon] = self.RM.renew.avbl_slot[:,i]
                cr_pt += self.pa.rep_horizon

            for act in self.eligible:
                if act is None:
                    cr_pt += self.pa.renewable + 2
                else:
                    compact_repr[cr_pt] = act.durations[0]
                    cr_pt += 1
                    for i in range(self.pa.renewable):
                        compact_repr[cr_pt] = act.re_requests[0][i]
                        cr_pt += 1
                    compact_repr[cr_pt] = act.after_duration
                    cr_pt += 1
            for act in self.backlog:
                compact_repr[cr_pt] = act.after_duration
                cr_pt += 1
                if self.pa.network_compact_dim == cr_pt:
                    break

            # print compact_repr
            return compact_repr


    def get_reward(self):

        reward = 0
        for j in self.RM.running_job:
            reward += self.pa.delay_penalty / float(j.durations[0])

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.durations[0])

        for j in range(self.pa.act_job):
            if not self.activities[j + 1].queued:
                reward += self.pa.dismiss_penalty / float(self.activities[j+1].durations[0])

        return reward

    def step(self, a):
        # need to change action for easily seeing
        self.stage += 1
        status = None
        done = False
        reward = 0
        info = None
        # later non resource check about shortage



        Allocated = True
        # print("env action: "+ str(a))
        if a == self.pa.num_queue:
            Allocated = False
            # reward = self.pa.delay_penalty * (self.eligible_cnt / self.pa.num_nw)
        elif self.eligible[a] == None :
            # reward = self.pa.delay_penalty * (self.eligible_cnt / self.pa.num_nw)
            Allocated = False
        else:
            # reward = self.RM.allocate_activity(self.activities[action])
            # print(self.eligible[a].no)
            Allocated = self.RM.allocate_activity(self.eligible[a], self.curr_time)
            # print(Allocated)
            if Allocated:
                # print('allocated: ' + str(self.eligible[a].no))
                # reward = len(self.allocated)
                # reward += len(self.allocated)
                # done = True
                # need to replay without this number

                self.allocated.append(self.eligible[a].no)
                # print(self.allocated)
                # reward += (len(self.allocated) - 1) * self.pa.sequence_reward
                # remove precedence of this activity at the others
                index = self.eligible[a].no - 2
                for act in self.activities:
                    if act.rep_precedence[0, index] == 1:
                        act.rep_precedence[0, index] = 0
                        if act.availableStart < self.eligible[a].finish_time - 1:
                            act.availableStart = self.eligible[a].finish_time - 1
                act = self.eligible[a]

                self.eligible[a] = None
                self.eligible_cnt -= 1

                if np.sum(self.activities[-1].rep_precedence) == 0:
                    done = True
                    endtime = 0
                    for act in self.activities: #change check precedence of end point
                        if endtime < act.finish_time:
                            endtime = act.finish_time
                    # print('all done')
                    # print('endTime: '+str(endtime))

                    # reward = 50 - endtime
                    # reward = self.pa.delay_penalty * endtime
                    # reward = float(endtime-1) / float(self.pa.dueDate) * self.pa.delay_penalty
                    reward = float(self.pa.opt[self.pa.file]) / float(endtime-1)
                    # print self.pa.file + ', opt: ' + str(self.pa.opt[self.pa.file]) + ', endtime:'+ str(endtime-1)

                    # reward = float(endtime-1) / float(self.pa.opt[self.pa.file])
                    # reward = (endtime - self.pa.opt[self.pa.file] - 1) * self.pa.delay_penalty
                    self.done_reward = endtime - self.pa.opt[self.pa.file] - 1
                    # print(self.done_reward)

        if Allocated is False:
            # print("????")
            self.curr_time += 1
            self.RM.time_proceed(self.curr_time)

            # reward = self.get_reward()

        self.findEligible(s_type=self.s_type)

        # for i in range(self.pa.act_job):
        #     print(str(self.activities[i+1].no)+': ', end='')
        #     print(self.activities[i+1].rep_precedence)
        # for act in self.eligible:
        #     if act is not None:
        #         print(str(act.no))
        ob = self.observe()

        if self.stage == self.pa.episode_max_length - 1:
            # reward = self.pa.episode_max_length - self.pa.opt[self.pa.file] - 1
            # reward = float(self.pa.opt[self.pa.file]) / float(self.pa.episode_max_length - 1)
            reward = float(self.pa.episode_max_length - 1) / float(self.pa.opt[self.pa.file]) * self.pa.delay_penalty

            self.done_reward = self.pa.episode_max_length - self.pa.opt[self.pa.file] - 1
            done = True
            # reward = -100

        return ob, reward, done, self.job_record


class JobSlot:
    def __init__(self, slot):
        self.slot = slot

class JobRecord:
    def __init__(self):
        self.record = {}

class ResourceManagement:
    def __init__(self, re, non, pa):
        self.renew = re
        self.nonRenew = non
        self.pa = pa
        self.running_job = []
    def allocate_activity(self, activity, curr_time):
        success = False
        # print('availableStart: ' + str(activity.availableStart))
        start = activity.availableStart - curr_time
        if start < 0:
            start = 0
        for t in range(start, self.pa.rep_horizon - activity.durations[0]):
            new_avbl_res = self.renew.avbl_slot[t: t + activity.durations[0], :] - activity.re_requests[0]

            if np.all(new_avbl_res[:] >= 0):

                self.renew.avbl_slot[t: t + activity.durations[0], :] = new_avbl_res

                for res in range(self.renew.num_res):
                    for i in range(t, t + activity.durations[0]):
                        avbl_slot = np.where(self.renew.forinput[res][i, :] == 1)[0]
                        self.renew.forinput[res][i, avbl_slot[: activity.re_requests[0][res]]] = 0

                activity.start_time = curr_time + t + 1 # start point of t is 0
                activity.finish_time = activity.start_time + activity.durations[0]
                # activity.rep_precedence[0, 0] = 1

                success = True
                self.running_job.append(activity)
                break

        return success
    def time_proceed(self, curr_time):
        self.renew.avbl_slot[: -1, :] = self.renew.avbl_slot[1:, :]
        self.renew.avbl_slot[-1, :] = self.renew.res_slot

        for i in range(self.renew.num_res):
            self.renew.forinput[i][:-1, :] = self.renew.forinput[i][1:, :]
            self.renew.forinput[i][-1, :] = 1
        for job in self.running_job:
            if job.finish_time <= curr_time:
                self.running_job.remove(job)



class RenewableResources:
    def __init__(self, pa):
        self.num_res = pa.renewable
        self.time_horizon = pa.rep_horizon
        self.res_slot = []
        for i in range(self.num_res):
            self.res_slot.append(pa.resourceAvailabilities[i])
        self.avbl_slot = np.empty(0).reshape(self.time_horizon, 0)
        self.forinput = []
        for i in range(self.num_res):
            temp = np.ones((self.time_horizon, 1))*self.res_slot[i]
            self.avbl_slot = np.hstack([self.avbl_slot, temp])

            rep_temp = np.ones((self.time_horizon, self.res_slot[i]))
            self.forinput.append(rep_temp)



class NonRenewableResources:
    def __init__(self, pa):
        self.num_res = pa.nonrenewable
        self.time_horizon = pa.rep_horizon
        self.res_slot = []
        for i in range(self.num_res):
            self.res_slot.append(pa.resourceAvailabilities[self.num_res+i])

        self.avbl_slot = np.empty(0).reshape(self.time_horizon,0)
        self.forinput = []
        for i in range(self.num_res):
            temp = np.ones((self.time_horizon,1))*self.res_slot[i]
            self.avbl_slot = np.hstack([self.avbl_slot, temp])

            rep_temp = np.ones((self.time_horizon, self.res_slot[i]))
            self.forinput.append(rep_temp)


class Activity:
    def __init__(self, pa, index):
        self.no = index + 1
        self.numOfModes = pa.modes[index]
        self.durations = []
        self.re_requests = []
        self.non_requests = []
        self.enter_time = 0
        self.start_time = -1
        self.finish_time = -1
        self.availableStart = -1
        self.queued = False


        for request in pa.requests[index]: # for multi mode
            self.durations.append(request[0])
            self.re_requests.append(request[1 : 1 + pa.renewable])
            self.non_requests.append(request[1 + pa.renewable : ])
        memory = np.ones((pa.jobs), dtype=int) * -1
        self.after_duration = self.recursive_successor_duration(pa, index, memory)
        # print("job: "+ str(self.no)+ ', after_duration: '+ str(self.after_duration))

        self.precedence = [] # update


        for i in range(pa.jobs-1):
            if i == 0:
                continue
            if i+1 != self.no:
                if self.no in pa.successors[i]:
                    self.precedence.append(i+1)
        if len(self.precedence) == 0:
            self.availableStart = 0
        self.rep_precedence = np.zeros((1, pa.act_job))

        for i in self.precedence:
            self.rep_precedence[0, i-2] = 1


        self.final_rep_precedence = self.rep_precedence.copy()
        if self.no == pa.jobs:
            return

        self.successors = pa.successors[index]

    def recursive_successor_duration(self, info, index, memory):
        # if len(info.successors[index]) == 0 :
        #     return 0;
        if index == info.jobs-1:
            return 0;
        # if memory[index] != -1:
        #     return memory[index]
        duration = info.requests[index][0][0]

        max = 0
        # print(index)
        # print(len(info.successors))
        for i in info.successors[index]:
            tmp = 0
            if memory[i - 1] != -1:
                tmp = memory[i - 1]
            else:
                tmp = self.recursive_successor_duration(info, i - 1, memory)
                memory[i - 1] = tmp
            memory[i - 1] = tmp

            if max < tmp:
                max = tmp
        return duration + max


if __name__ == '__main__':
    # info = fp.parser('j102_2.mm')
    pa = pa.Parameters()

    env = Env(pa, repre='compact')
    env.observe()
    # print()
    # env.step(3)  # 5
    # env.step(0) # 2
    # env.step(1) # 3

    _1, _2, _3, _4 =env.step(0) # 2
    # _1, _2, _3, _4 = env.step(4)  # 2

    # env.step(3) # 5
    # env.step(4) # 6
    # _1, _2, _3, _4 =env.step(1)  #3

    # _1, _2, _3, _4 = env.step(2)
    # _1, _2, _3, _4 = env.step(0)
    # _1, _2, _3, _4 = env.step(3)
    # _1, _2, _3, _4 = env.step(0)
    # _1, _2, _3, _4 = env.step(1)
    # _1, _2, _3, _4 = env.step(0)
    # _1, _2, _3, _4 = env.step(1)
    # _1, _2, _3, _4 = env.step(2)
    # _1, _2, _3, _4 = env.step(0)
    # _1, _2, _3, _4 = env.step(1)
    # _1, _2, _3, _4 = env.step(1)





