

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
        self.isTest = False
        self.fp = None

    def setTest(self, flag):
        self.isTest = flag

    def setFP(self, fp):
        self.fp = fp

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

        if self.isTest:
            data = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (np.mean(rews), np.max(rews), np.min(rews), str(float(np.sum(list[:1])) / float(np.sum(list))),
                                str(float(np.sum(list[:2])) / float(np.sum(list))), str(float(np.sum(list[:3])) / float(np.sum(list))),
                                str(float(np.sum(list[:4])) / float(np.sum(list))), str(float(np.sum(list[:8])) / float(np.sum(list))),
                                np.mean(times), np.std(times), np.max(times), np.min(times))

            if self.fp is not None:
                self.fp.write(data)
            print data

            # print "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % (np.mean(rews), np.max(rews), np.min(rews), str(float(np.sum(list[:1])) / float(np.sum(list))),
            #                     str(float(np.sum(list[:2])) / float(np.sum(list))), str(float(np.sum(list[:3])) / float(np.sum(list))),
            #                     str(float(np.sum(list[:4])) / float(np.sum(list))), str(float(np.sum(list[:8])) / float(np.sum(list))),
            #                     np.mean(times), np.std(times), np.max(times), np.min(times))

        else:
            print "Elapsed time\t %s" % str((timer_end - timer_start) / np.sum(list)), "seconds"
            print "T-Reward Mean\t %s" % np.mean(rews)
            print "T-endtime Mean\t %s" % np.mean(times)

            # print rews

            print "T-Opt\t\t %s" % str(float(np.sum(list[:1])) / float(np.sum(list)))
            print "T-1ap\t\t %s" % str(float(np.sum(list[:2])) / float(np.sum(list)))
            print "T-2ap\t\t %s" % str(float(np.sum(list[:3])) / float(np.sum(list)))
            print "T-3ap\t\t %s" % str(float(np.sum(list[:4])) / float(np.sum(list)))
            print "T-7ap\t\t %s" % str(float(np.sum(list[:8])) / float(np.sum(list)))



        # print float(np.sum(list[:8]))/float(np.sum(list))

if __name__ == '__main__':
    import sys
    import parameters_RCPSP
    from basic import basic_RCPSP_env as environment
    # info = fp.parser('data/raw/j102_4.mm.backup')
    if len(sys.argv) > 2:
        file = sys.argv[1]
        s_type = sys.argv[2]
    else:
        file = 'data/result/onebestitem/'
        s_type = 'Max'
    pa = parameters_RCPSP.Parameters()
    pa.s_type = s_type
    fp = open(file+'result_'+s_type+'.txt', 'w')
    env = environment.Env(pa)

    ###################################
    print "iter\tr-mean\tr-max\tr-min\topt\t1ap\t2ap\t3ap\t7ap\td-mean\td-std\td-max\td-min"
    fp.write("iter\tr-mean\tr-max\tr-min\topt\t1ap\t2ap\t3ap\t7ap\td-mean\td-std\td-max\td-min\n")
    ttp = TestingTrainParameter(pa, env)
    ttp.setTest(True)
    ttp.setFP(fp)
    for i in range(0, 3001, 50):
        print str(i) + '\t',
        fp.write(str(i) + '\t')
        ttp.reset(pa, env, learner=None, pg_resume=file + 'qs5_hsize20_' + str(i) + '.pkl')
        ttp.start()
    ###################################
    fp.close()
    # ttp = TestingTrainParameter(pa, env, learner=None, pg_resume='data/result/onebestitem/qs5_hsize20_2100.pkl')
    # ttp.start()
