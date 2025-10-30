import csv

import pandas as pd

from scipy.io import loadmat

from process.dropDetect import dropdetect
from process.generateGt import generate_gt
from process.preProcessVital import preProcessVital


def processData():
    fps = 40
    fps_bre = 20
    str_second = 0
    en_second = 10
    str_ = str_second * fps
    en = fps * en_second
    t = 10
    count = ((en - str_) // fps) // t
    cut_sample = t * fps
    drop_t = 10
    cut_drop = drop_t * fps
    sample_uwb = fps * count * t
    sample_bre = fps_bre * count * t
    resolution = fps / cut_sample
    rangeK = 15

    CIRMatrix = loadmat('')
    CIRMatrix = CIRMatrix['CIRMatrix']

    BreathData = []
    with open('', mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            BreathData.append(row)

    TimeStamps = pd.read_csv('')
    TimeStamps = TimeStamps['0'].tolist()

    Breath_sig_final = generate_gt(CIRMatrix, BreathData, TimeStamps)
    [dropIndex, drop_sig_r, drop_sig_p] = dropdetect(CIRMatrix)
    if len(Breath_sig_final) >= sample_uwb:
        [breath_gt, record_I1, record_Q1, record_P1, record_I2, record_Q2, record_P2] = preProcessVital(str_, en,
                                                                                                        cut_sample,
                                                                                                        CIRMatrix,
                                                                                                        Breath_sig_final)
    return


if __name__ == '__main__':
    processData()
