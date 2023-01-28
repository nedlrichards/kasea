import argparse
from kasea import Ping

parser = argparse.ArgumentParser(
                    prog = 'kasea',
                    description = 'executes experiment specified in toml file')
parser.add_argument('filename')

args = parser.parse_args()

ping = Ping(args.filename)

[ping.one_time(i) for i in range(ping.xmission.num_steps)]
