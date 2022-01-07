import os
import numpy as np
import logging
import time



def output_log(filename):
    logging.basicConfig(
            level=logging.NOTSET,  # 定义输出到文件的log级别，
            format='%(asctime)s || %(message)s',  # 定义输出log的格式
            datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
            filename=filename,  # log文件名
            filemode='a')  # 写入模式“w”或“a”
    console = logging.StreamHandler()  # 定义console handler
    console.setLevel(logging.NOTSET)  # 定义该handler级别
    formatter = logging.Formatter('%(asctime)s || %(message)s')  # 定义该handler格式
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)  # 实例化添加handler
    return logging


# Thread 7 Simulation finesed in 5.988238s

class BsimAnalysis():
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def search_time(self):
        dict_all_file_time = {}
        for file_name in os.listdir(self.dir_path):
            file_path = os.path.join(self.dir_path, file_name)
            log.info('***********************开始提取{}文件**********************'.format(file_path))
            r = os.popen("grep -rn 'Simulation finesed' {}".format(file_path))
            log.info('------------------------------------------------')
            result = r.read()
            log.info(result)
            log.info(type(result))
            r.close()
            log.info('------------------------------------------------')
            text_list = [i for i in result.split('\n') if len(i) > 5]
            start_line_num = 0
            list_all = []
            list1 = []
            for i in text_list:
                line = int(i.split(':')[0])
                time_s = i.split(' ')[-1].replace('s', '')
                if line - start_line_num > 1 and list1 != []:
                    list_all.append(list1)
                    list1 = [float(time_s)]
                else:
                    list1.append(float(time_s))
                start_line_num = line
            list_all.append(list1)
            list_all_time = [i for i in list_all if i != []]
            num_list = []
            for i in list_all_time:
                num_list.append(len(i))
                log.info(i)
            log.info('一共找到{}组数据，每组数据个数分别为{}。'.format(len(list_all_time), num_list))
            file_list = []
            for list1 in list_all_time:
                file_list.append(max(list1))
            # a = np.array(list_all_time)
            name = '_'.join(file_name.split('_')[:-1])
            if name in dict_all_file_time:
                a = dict_all_file_time[name]
                print(a)
                dict_all_file_time[name] = np.r_[a,np.array([file_list])]
            else:
                dict_all_file_time[name] = np.array([file_list])
        return dict_all_file_time
        

if __name__ == '__main__':
    log = output_log('./bsim_analysis.log')
    dir_path = r'/archive/share/east8411/exercise/run_all'
    bsimanalysis = BsimAnalysis(dir_path)
    time_list = bsimanalysis.search_time()
    # log.info(str(time_list))

    for key in time_list:
        time_list[key] = np.sort(time_list[key], axis=0)
        time_list[key] = time_list[key][0:7]
        time_list[key] = time_list[key].mean(axis=0)

    log.info(str(time_list))

