import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

color = dict()
color["HNSW"] = '#80BFFF'
color["FLAT"] = '#0072E5'

mk = dict()
mk["HNSW"] = 'o'
mk["FLAT"] = 's'

style = dict()
style["HNSW"] = '-'
style["FLAT"] = '-.'

# baseline&datafile
baselines = ["HNSW", "FLAT"]
collections = ["audio"]
prefix = "log"
suffix = ".txt"

# dataset
dataset = ["YT-Audio"]

def make_interval(sorted_recalls, sorted_runtimes):
    recall_dict = dict()
    for i, recall in enumerate(sorted_recalls):
        discrete_recall = int(recall*10)
        if discrete_recall in recall_dict:
            tot, cnt = recall_dict[discrete_recall]
            recall_dict[discrete_recall] = [tot+sorted_runtimes[i], cnt+1]
        else:
            recall_dict[discrete_recall] = [sorted_runtimes[i], 1]
    tmpList = []
    for discrete_recall, item in recall_dict.items():
        tmpList.append([discrete_recall, item[0]/item[1]])
    sorted_tmpList = sorted(tmpList, key=lambda x: x[0])

    recalls, runtimes = [], []
    for recall, runtime in sorted_tmpList:
        recalls.append(recall/10.0)
        runtimes.append(runtime)

    return recalls, runtimes

# load log
def ReadResult(filename):
    recalls = []
    runtimes = []
    with open(filename, "r") as fin:
        for i,line in enumerate(fin):
            line = line.strip()
            line = line[1:len(line)-1]
            tmpList = line.split(", ")
            tmpList = list(map(float, tmpList))
            if i==0:
                runtimes = tmpList
            else:
                recalls = tmpList
    assert(len(recalls) == len(runtimes))
    performances = [list(item) for item in zip(recalls, runtimes)]
    sorted_performances = sorted(performances, key=lambda x: x[0])

    sorted_recalls = [item[0] for item in sorted_performances]
    sorted_runtimes = [item[1] for item in sorted_performances]

    sorted_recalls, sorted_runtimes = make_interval(sorted_recalls, sorted_runtimes)

    return sorted_recalls, sorted_runtimes

def PlotFigure():
    plt.rcParams['font.family'] = 'Arial'
    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots(1, 1)

    for method in baselines:
        filename = f"{method}.log"
        Recall, Runtime = ReadResult(filename)
        m = method
        lw, ms, mew = 3, 9, 3
        ax.plot(Runtime, Recall, color=color[m], label=m, marker=mk[m], lw=lw, ms=ms, mew=mew, mec=color[m],
                                mfc='none', linestyle=style[m])
    ax.set_title("YT-Audio", fontsize=16, color='k')
    ax.set_xlabel('Search Time (ms)', fontsize=16, color='k')
    ax.set_xscale('log')
    ax.set_ylabel('Recall', fontsize=16, color='k')
    ax.legend(baselines, fontsize=12)
    fig.tight_layout()
    plt.savefig("result.png")
    plt.show()

if __name__ == "__main__":
    PlotFigure()