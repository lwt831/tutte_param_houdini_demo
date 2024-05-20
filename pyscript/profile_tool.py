from line_profiler import LineProfiler


# usage:
# @Profile(follow=[SubFunc1, SubFunc2])
# def Func(args0):
#   ...
#   SubFunc1(args1)
#   ...
#   SubFunc2(args2)
#   ...
def Profile(follow=[]):
    def Inner(Func):
        def ProfiledFunc(*arg, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(Func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return Func(*arg, **kwargs)
            finally:
                profiler.print_stats()

        return ProfiledFunc

    return Inner
