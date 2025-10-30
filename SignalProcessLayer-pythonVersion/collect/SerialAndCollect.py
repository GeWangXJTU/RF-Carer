import csv
import threading
import time

import numpy as np
import pandas as pd
import scipy.io as sio

from collect import uwb_data_cache, bre_data_cache, UWB_data_collect
from collect.time_cache import time_queue
from collect.utils import FRAMES, OFFSET



def plot():
    print("start plot")
    frecv = True
    uwb_datas = None
    bre_datas = []
    time_stamps = []
    while True:
        if uwb_data_cache.uwb_data_queue.empty() or bre_data_cache.bre_data_queue.empty():
            time.sleep(0.01)
            continue
        else:
            uwb_data = uwb_data_cache.uwb_data_queue.get()
            bre_data = bre_data_cache.bre_data_queue.get()
            bre_datas.append(bre_data)
            time_data = time_queue.get()
            time_stamps.append(time_data)
            uwb_data = uwb_data.reshape(1, -1, 2)
            if frecv:
                uwb_datas = uwb_data
                frecv = False
            else:
                uwb_datas = np.vstack((uwb_datas, uwb_data))

            datas_len = uwb_datas.shape[0]
            if datas_len >= FRAMES:
                iq = uwb_datas[-FRAMES:, :, :]
                org = iq[:, OFFSET:, 0] + 1j * iq[:, OFFSET:, 1]
                x = org[:, :]
                file_name = time.strftime("%Y%m%d_%H%M%S.mat", time.localtime())
                sio.savemat('{}'.format(file_name), {'CIRMatrix': x})
                uwb_datas = uwb_datas[FRAMES:, :, :]

            if len(bre_datas) >= FRAMES:
                file_name = time.strftime("%Y%m%d_%H%M%S.csv", time.localtime())
                file_path = f'{file_name}'
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    for row in bre_datas[:FRAMES]:
                        writer.writerow(row)
                bre_datas = bre_datas[FRAMES:]

            if len(time_stamps) >= FRAMES:
                df = pd.DataFrame(time_stamps[:FRAMES])
                file_name = time.strftime("%Y%m%d_%H%M%S.csv", time.localtime())
                file_path = '{}'.format(file_name)
                df.to_csv(file_path, index=False)
                time_stamps = time_stamps[FRAMES:]
    return


def main():
    uwb = UWB_data_collect.SerialCollect("COM6", "COM7")
    if not uwb.state:
        print('serial init error.')
        exit(0)
    collectuwb = threading.Thread(target=uwb.recv)
    collectuwb.setDaemon(True)
    collectuwb.start()

    try:
        plot()
    except KeyboardInterrupt:
        print("Stopping...")
        collectuwb.join()
        exit(0)


if __name__ == '__main__':
    main()
