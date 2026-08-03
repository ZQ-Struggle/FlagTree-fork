# Proton C++ tests moved to FlagPrism; retain the legacy target for audit only.
#[[
add_mlir_library(TritonTestProton
  TestScopeIdAllocation.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
]]
