import os
import sys
import datetime
from tools import report_res, clear_log, parse_config, get_time_str
from work import do_work


def main():
    config_path = os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'
    cfg, cfg_str = parse_config(config_path)
    if not cfg['log']['display']:
        sys.stdout = open(os.devnull, 'w')
    clear_log(cfg)

    start_time = datetime.datetime.now()
    time_str = 'start time: {}\n'.format(start_time.strftime('%Y-%m-%d %H:%M:%S'))
    res = do_work(cfg)
    end_time = datetime.datetime.now()
    time_str += 'end time: {}\n'.format(end_time.strftime('%Y-%m-%d %H:%M:%S'))
    time_str += 'used time: {}\n'.format(get_time_str(end_time - start_time))

    report_res(cfg_str + '\n' + time_str + '\n' + res)


if __name__ == '__main__':
    main()
