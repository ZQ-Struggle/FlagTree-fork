add_subdirectory(Analysis)
add_subdirectory(Dialect)
add_subdirectory(Instrumentation)
# Proton tests moved to FlagPrism; the legacy parent target is intentionally disabled.
#[[
add_subdirectory(Proton)
]]
