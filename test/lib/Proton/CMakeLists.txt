# Profiler C++ tests moved to FlagPrism; retain this internal target for audit.
#[[
add_mlir_library(TritonTestProton
  TestScopeIdAllocation.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
]]
