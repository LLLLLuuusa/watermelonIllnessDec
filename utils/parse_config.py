# ================================================================
#
#   Editor      : Pycharm
#   File name   : parse_config
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-4 14:05
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : 根据路径解析data文件
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================

def parse_data_cfg(path):
    """Parses the data configuration file"""
    options = dict()
    # options['gpus'] = '0,1,2,3'
    # options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options