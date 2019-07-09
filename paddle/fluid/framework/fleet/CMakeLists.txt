if(WITH_PSLIB)
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper scope pslib_brpc pslib)
    cc_library(fleet_geo_wrapper SRCS fleet_geo_wrapper.cc DEPS framework_proto variable_helper scope pslib_brpc pslib pslib_mlparallel pslib_mpi pslib_mpicxx pslib_mpiopenrte pslib_mpiopenpal)
else()
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper scope)
    cc_library(fleet_geo_wrapper SRCS fleet_geo_wrapper.cc DEPS framework_proto variable_helper scope)
endif(WITH_PSLIB)

cc_library(nccl_wrapper SRCS nccl_wrapper.cc DEPS framework_proto variable_helper scope)
