
from src.utils import mkdirs_safe, print_fl
import os
import sys
import time
from glob import glob


def child_done(name, parent_watch_dir, child_name):
    # write child done file
    watch_dir = parent_watch_dir + '/' + name
    write_path = ('%s/child_%s.watch' % (watch_dir, str(child_name)))
    print_fl(write_path)
    with open(write_path, 'wb') as f:
        f.write('Done')

class TaskDriver:
    """
    Class to split up work across multiple processes using files written to
    a watch directory to signal work completion.
    """

    def __init__(self, name, parent_watch_dir, num_wait, sleep_time=600, 
        timer=None):

        self.name = name
        self.watch_dir = parent_watch_dir + '/' + name
        self.num_wait = num_wait
        self.sleep_time = sleep_time
        self.timer = timer

        # create directory to watch for slurm jobs to finish
        mkdirs_safe([self.watch_dir])

    def print_driver(self):
        print_fl("Using Task Driver for %s in directory %s. %d children." % 
                 (self.name, self.watch_dir, self.num_wait))

    def cleanup(self):
        if len(self.watch_dir) == 0: 
            raise ValueError("Invalid watch directory: %s" % self.watch_dir)

        files = glob('%s/child_*.watch' % self.watch_dir)
        for f in files:
            os.remove(f)

    def wait_for_tasks(self):
        """
        Wait for the number of child processes to finish by monitoring watch
        directory
        """

        while True:

            # count # of child processes done
            files = os.listdir(self.watch_dir)
            num_done = sum([f.endswith('.watch') for f in files])
            print_fl("%d/%d finished. " % (num_done, self.num_wait), end='')

            if self.timer is not None:
                print_fl(" %s " % self.timer.get_time(), end='')

            # number of children done
            if num_done == self.num_wait:
                print_fl("All children done.")
                break

            # sleep until all children are done
            else:
                print_fl("Sleeping %ds" %
                    (self.sleep_time))
                time.sleep(self.sleep_time)

        # clean up watch directory
        print_fl("Cleaning up watch directory.")
        self.cleanup()
