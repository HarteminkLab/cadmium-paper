
import time
from src.utils import print_fl


class Timer:

    start_time = 0.0

    def __init__(self):
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.elapsed_time = time.time() - self.start_time

    def get_time(self):
        self.elapsed_time = time.time() - self.start_time
        hours, rem = divmod(self.elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    def print_time(self, indent=0):
        indentspace = ' ' * indent
        print_fl("%s%s" % (indentspace, self.get_time()))

    def print_label(self, label, indent=0):
        indentspace = ' ' * indent
        print_fl("%s%s - %s" % (indentspace, label, self.get_time()))


class TimingContext(object):

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, typ, value, traceback):
        self.elapsed_time = self.timer.get_time()

    def get_time(self):
        return self.timer.get_time()
