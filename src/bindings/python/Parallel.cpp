#include "python_bindings.h"
#include "Parallel.h"

void init_parallel(py::module_& m) {
    tensorplay::parallel::init_num_threads();

    m.def("set_num_threads", &tensorplay::parallel::set_num_threads,
          py::arg("nthreads"),
          "Sets the number of threads used for intraop parallelism on CPU.");

    m.def("get_num_threads", &tensorplay::parallel::get_num_threads,
          "Returns the number of threads used for parallelizing CPU operations");

    m.def("get_thread_num", &tensorplay::parallel::get_thread_num,
          "Returns the current thread number (starting from 0) in the current "
          "parallel region, or 0 in the sequential region");

    m.def("in_parallel_region", &tensorplay::parallel::in_parallel_region,
          "Checks whether the code runs in a parallel region");

    m.def("get_parallel_info", &tensorplay::parallel::get_parallel_info,
          "Returns a detailed string describing parallelization settings");
}