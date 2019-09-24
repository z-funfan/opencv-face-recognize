# 最粗暴的客流统计

# 统计每小时客流，本小时内，发现新人则计数+1
# 日期变换，所有记录清零
# 同一张人脸，在一分钟之内不重复计数
import time

INIT_COUNTS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
                11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
daily_counts = INIT_COUNTS
ref_date = 20190101
ref_hour = 0
last_update_dict = {}

def initData():
    global daily_counts, ref_date, ref_hour, last_update_dict
    daily_counts = INIT_COUNTS
    ref_date, ref_hour = map(int, time.strftime("%Y%m%d %H").split())
    last_update_dict = {}

def garbageDataByDate():
    current_date, current_hour = map(int, time.strftime("%Y%m%d %H").split())
    if (current_date != ref_date):
        # 一旦日期切换，清空所有数据
        print('Date changed, clean all data')
        initData()
    elif (current_hour != ref_hour):
        # 每小时清空人脸记录，但不清计数
        print('Hour changed, clean last_update_dict')
        last_update_dict.clear()
    else:
        # do nothing
        pass
    return current_hour

def updateData(name, hour, time):
    last_update_dict[name] = time
    daily_counts[hour] += 1
    print("{} detected, current hour is {}, count is {}".format(name, hour, daily_counts[hour]))

# 如果一分钟之内没有重复检测，计数加一，并更新时间
def faceDetected(name):
    now = int(time.time())
    hour = garbageDataByDate()
    if (name in last_update_dict):
        lastUpdateTime = last_update_dict[name]
        # 一分钟之内用一张人脸不重复计数
        if ((now - lastUpdateTime) > 60):
            updateData(name, hour, now)
    else:
        # 发现新的客户，更新计数
        updateData(name, hour, now)

def getCounts():
    return daily_counts

initData()

if __name__ == '__main__':
    while True:
        faceDetected('fengfan_zheng')
        time.sleep(20)
