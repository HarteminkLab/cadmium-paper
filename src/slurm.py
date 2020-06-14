
from src.utils import run_cmd, print_fl

def submit_sbatch(exports, script, directory):
    # spawn child sbatch jobs
    command = ('sbatch -D %s --export=%s %s' %
               (directory, exports, script))
    print_fl("Submiting job: %s " % command)
    run_cmd(command)

