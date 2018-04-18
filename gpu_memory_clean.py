import os
import psutil

output = os.popen('fuser /dev/nvidia*', 'r')
pids = set(output.read().split())

for pid in pids:
    p = psutil.Process(int(pid))
    if p.ppid() == 1 and p.cpu_times().system < 1:
        p.kill()

