from kasea import Ping


ping = Ping('experiments/flat.toml', ier='FT')
%time ping.one_time(0)
