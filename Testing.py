

import pg_network
import cPickle
import numpy as np
import time

class TestingTrainParameter:
    def __init__(self, pa, env, learner=None, pg_resume=None):
        if learner is None:
            self.pg_learner = pg_network.PGLearner(pa)
        else:
            self.pg_learner = learner

        if pg_resume is not None:
            net_handle = open(pg_resume, 'rb')
            net_params = cPickle.load(net_handle)
            self.pg_learner.set_net_params(net_params)
        # print("??")
        self.env = env
        self.pa = pa

    def reset(self, pa, env, learner=None, pg_resume=None):
        if learner is None:
            self.pg_learner = pg_network.PGLearner(pa)
        else:
            self.pg_learner = learner

        if pg_resume is not None:
            net_handle = open(pg_resume, 'rb')
            net_params = cPickle.load(net_handle)
            self.pg_learner.set_net_params(net_params)

        self.env = env
        self.pa = pa

    def start(self):
        self.env.test_type = 'test'
        # print("1")
        finish = self.env.reset()
        # print(finish)
        rews = []
        times = []
        timer_start = time.time()
        while finish:
            # print("2")

            ob = self.env.observe()
            for _ in xrange(self.pa.episode_max_length):

                a = self.pg_learner.choose_action_testing(ob)

                ob, rew, done, info = self.env.step(a)

                if done:
                    rews.append(rew)
                    times.append(self.env.done_reward)
                    # if len(rews)%10==0:
                    #     for index in range(len(info.record)):
                    #         print("no: " + str(info.record[index].no) + ", start: " + str(
                    #             info.record[index].start_time) + ", finish: " + str(info.record[index].finish_time))
                    # if rew>=20:
                    #     print self.env.pa.path + ', rews: '+str(rew)

                    break

            finish = self.env.reset()
        timer_end = time.time()
        list = np.bincount(times)
        # print list
        print "Elapsed time\t %s" % str((timer_end - timer_start)/np.sum(list)), "seconds"
        print "T-Reward Mean\t %s" % np.mean(rews)
        print "T-endtime Mean\t %s" % np.mean(times)

        # print "%s\t%s\t%s\t%s" % (np.mean(rews), np.max(rews), np.min(rews), np.std(rews))
        # print rews


        print "T-Opt\t\t %s" % str(float(np.sum(list[:1])) / float(np.sum(list)))
        print "T-1ap\t\t %s" % str(float(np.sum(list[:2])) / float(np.sum(list)))
        print "T-2ap\t\t %s" % str(float(np.sum(list[:3])) / float(np.sum(list)))
        print "T-3ap\t\t %s" % str(float(np.sum(list[:4])) / float(np.sum(list)))
        print "T-7ap\t\t %s" % str(float(np.sum(list[:8])) / float(np.sum(list)))
        # print float(np.sum(list[:8]))/float(np.sum(list))

if __name__ == '__main__':
    import parameters_RCPSP
    from basic import basic_RCPSP_env as environment
    # info = fp.parser('data/raw/j102_4.mm.backup')
    pa = parameters_RCPSP.Parameters()
    env = environment.Env(pa)

    ###################################
    # print "iter\tmean\tmax\tmin\tstd"
    # ttp = TestingTrainParameter(pa, env)
    # for i in range(300, 5000, 50):
    #     print str(i) + '\t',
    #     ttp.reset(pa, env, learner=None, pg_resume='data/preced30_5_hsize20_hori150_' + str(i) + '.pkl')
    #     ttp.start()
    ###################################

    ttp = TestingTrainParameter(pa, env, learner=None, pg_resume='data/result/oneitemqueue/onequeue_qs5_hsize20_450.pkl')
    ttp.start()
