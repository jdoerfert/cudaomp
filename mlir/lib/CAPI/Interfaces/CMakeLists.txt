add_mlir_public_c_api_library(MLIRCAPIInterfaces
  Interfaces.cpp

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface)
