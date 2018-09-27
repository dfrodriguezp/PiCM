import cProfile, pstats, io
import main

pr = cProfile.Profile()
pr.enable()
main.main()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("filename")
ps.print_stats()
print(s.getvalue())