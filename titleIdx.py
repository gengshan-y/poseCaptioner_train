import sys
import json
print(json.load(open('/home/gengshan/public_html/data/ori_data_list.json',\
                     'r'))[sys.argv[1]])
