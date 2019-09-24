

class File_Info:
    def __init__(self):
        self.projects = 0
        self.jobs = 0
        self.horizon = 0
        self.renewable = 0
        self.nonrenewable = 0
        self.doubly_constrained = 0
        self.releaseDate = 0
        self.dueDate = 0
        self.maxRes_Avbl = 0
        self.maxRes_Req = 0

        self.modes = []
        self.successors = []
        self.requests = []

        self.resourceAvailabilities = []


def opt_parser(path):
    f = open(path, 'r')
    opt = {}
    for line in f:
        res = [int(i) for i in line.split() if i.isdigit()]
        key = 'j30'+str(res[0])+'_'+str(res[1])+'.sm'
        # print(key)
        opt[key] = res[2]
    return opt

def get_list(path):
    f = open(path, 'r')
    list = []
    for file in f:
        list.append(file.strip())
    return list
# print(opt_parser('j30opt.sm'))


def parser(path):
    # print("path: " + path)
    file_info = File_Info()
    f = open(path,'r')
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()

    line = f.readline()
    tmp = line.split(':')
    file_info.projects = int(tmp[1].strip())

    line = f.readline()
    tmp = line.split(':')
    file_info.jobs = int(tmp[1].strip())

    line = f.readline()
    tmp = line.split(':')
    file_info.horizon = int(tmp[1].strip())

    line = f.readline()

    line = f.readline()
    tmp = line.split(':')
    file_info.renewable = int(tmp[1].strip().split(' ')[0])

    line = f.readline()
    tmp = line.split(':')
    file_info.nonrenewable = int(tmp[1].strip().split(' ')[0])

    line = f.readline()
    tmp = line.split(':')
    file_info.doubly_constrained = int(tmp[1].strip().split(' ')[0])

    line = f.readline()
    line = f.readline()
    line = f.readline()

    line = f.readline()
    tmp = line.split('     ')

    file_info.releaseDate = int(tmp[2].strip())

    file_info.dueDate = int(tmp[3].strip())
    # print("dudate:")
    # print file_info.dueDate
    line = f.readline()
    line = f.readline()
    line = f.readline()
    # print(line)
    for i in range(file_info.jobs):
        line = f.readline()
        # print(line)
        item = line.split('     ')
        file_info.modes.append(int(item[1].strip()))
        # print(int(item[1].strip()))
        if i == file_info.jobs-1:
            break
        tmp_successor = []
        # print(item[5].split('  '))
        for act in item[5].split('  '):
            tmp_successor.append(int(act.strip()))
        file_info.successors.append(tmp_successor)
        # print(tmp_successor)
        # print(i)
        # print(item[5])

    # line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()

    for i in range(file_info.jobs):

        modes = file_info.modes[i]

        activity = []
        line = f.readline()

        item = line.split('      ')
        # print(item)
        row = []

        duration = int(item[1].split()[1])

        row.append(duration)
        demand = item[2].split()
        # print(demand)
        for req in range(4):
            row.append(int(demand[req]))
            if file_info.maxRes_Req < int(demand[req]):
                file_info.maxRes_Req = int(demand[req])

        activity.append(row)
        for mode in range(modes-1):
            line = f.readline()
            item = line.split('      ')
            row1 = []
            duration = int(item[1].split()[1])
            row1.append(duration)
            demand = item[2].split()
            for req in range(4):
                row1.append(int(demand[req]))
                if file_info.maxRes_Req < int(demand[req]):
                    file_info.maxRes_Req = int(demand[req])
            activity.append(row1)


        file_info.requests.append(activity)
    # print(len(file_info.requests))
    # print(file_info.requests)
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()

    item = line.split('  ')

    # print(item)
    skip = 0
    for i in range(file_info.renewable + file_info.nonrenewable):
        if item[i+1+skip] == '':
            # print("skip?, i: "+ str(i+1))
            skip += 1
        # print(i+1+skip)
        file_info.resourceAvailabilities.append(int(item[i+1+skip]))
        # file_info.resourceAvailabilities.append(int(10))
        if file_info.maxRes_Avbl < int(item[i+1+skip]):
            file_info.maxRes_Avbl = int(item[i+1+skip])
    # print(file_info.resourceAvailabilities)



    f.close()


    return file_info
#
#
# def recursive_successor_duration(info, index, memory):
#     # if len(info.successors[index]) == 0 :
#     #     return 0;
#     if index == 11:
#         return 0;
#     # if memory[index] != -1:
#     #     return memory[index]
#     duration = info.requests[index][0][0]
#
#     max = 0
#     for i in info.successors[index]:
#         tmp = 0
#         if memory[i - 1] != -1:
#             tmp = memory[i - 1]
#         else:
#             tmp = recursive_successor_duration(info, i - 1, memory)
#             memory[i - 1] = tmp
#
#         if max < tmp:
#             max = tmp
#     return duration + max
#
#
# info = parser('../data/RCPSP/train/j1042_1.mm')
# memory = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
# duration = recursive_successor_duration(info, 0, memory)
# print(duration)
# # # print(memory)
# # info = parser('j102_2.mm')
# # a = [info, info]
# # print(a)
# # a.remove(info)
# # print(a)
# #
# #
# #
