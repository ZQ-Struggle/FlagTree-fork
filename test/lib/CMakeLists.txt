add_subdirectory(Analysis)
add_subdirectory(Dialect)
add_subdirectory(Instrumentation)
# Profiler tests moved to FlagPrism; this legacy parent target stays disabled.
#[[
add_subdirectory(Proton)
]]
