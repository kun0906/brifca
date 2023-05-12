
from process_runner import *

parser = argparse.ArgumentParser()
parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
                    action='store_true', help='force')
parser.add_argument("--max-procs", type=int, default=-1)  # -1 for debugging
parser.add_argument("--arr-size", type=int, default=-1)
parser.add_argument("--arr-index", type=int, default=-1)
parser.add_argument("-n", type=int, default=50)
parser.add_argument("-d", type=int, default=2)
parser.add_argument("--update_method", type=str, default='mean')
args = parser.parse_args()



print(args)
