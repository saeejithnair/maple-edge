from datetime import datetime as dt
import os
from multiprocessing import Process
import signal
import subprocess
import time


class PerfRecorder:
    def __init__(self):
        self.perf_stats = None
        self.start_time = None
        self.date_format = "%a %b %-m %-H:%M:%S %Y"

    def start_profiling(self):
        self.start_time = dt.now()

        perf_ops = "cpu-cycles,instructions,cache-references,cache-misses," \
                   "L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses," \
                   "LLC-loads,LLC-store-misses,LLC-stores,cpu-migrations"

        self.perf_process = subprocess.Popen(
            ['perf', 'stat',  '-p', str(os.getpid()), '-x', ',',
             '-e', perf_ops], stderr=subprocess.PIPE)

    def terminate_perf(self):
        time.sleep(1)  # wait until parent runs `stderr.read()`
        self.perf_process.send_signal(signal.SIGINT)
        exit(0)

    def stop_profiling(self):
        p = Process(target=self.terminate_perf)
        p.start()
        self.perf_stats = self.perf_process.stderr.read().decode(
                                "utf-8").rstrip('\n')
        p.join()

    def get_stats(self):
        if self.perf_stats is None:
            raise ValueError(
                "ERROR, no stats available. If profiling has been started, "
                "call stop_profiling to obtain perf stats.")

        start_time = self.start_time.strftime(self.date_format)
        perf_stats = f"# started on {start_time} " \
                     f"Perf Subprocess\n\n{self.perf_stats}"
        return perf_stats
