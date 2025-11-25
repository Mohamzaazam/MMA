import cProfile
import pstats
from pstats import SortKey

p = pstats.Stats("profile.prof")
p.sort_stats("cumulative").print_stats(20)