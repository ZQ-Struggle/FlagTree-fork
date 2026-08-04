# FlagPrism supplies ProtonIR when the bundled tools are enabled.
if(TRITON_BUILD_FLAGPRISM)
  set(_ProtonIRLib ProtonIR)
endif()

if(FLAGTREE_TLE)
  set(_TleIRLib TleIR)  # flagtree tle raw
endif()

add_triton_library(TritonToTritonGPU
    RelayoutTritonGPU.cpp
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    ${_ProtonIRLib}
    TritonGPUIR
    ${_TleIRLib}
)
