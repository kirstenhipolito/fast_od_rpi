Last login: Mon Jun  1 15:50:54 on ttys001

The default interactive shell is now zsh.
To update your account to use zsh, please run `chsh -s /bin/zsh`.
For more details, please visit https://support.apple.com/kb/HT208050.
kirstenhipolito_EEE@Kirstens-MacBook-Pro:~/virtualenvs/EEE_198$ docker ps
CONTAINER ID        IMAGE                              COMMAND             CREATED              STATUS              PORTS               NAMES
46717ebbb358        docker_lce_cross_compiler:latest   "/bin/bash"         About a minute ago   Up About a minute                       larq_xcomp
kirstenhipolito_EEE@Kirstens-MacBook-Pro:~/virtualenvs/EEE_198$ docker stop larq_xcomp
larq_xcomp
kirstenhipolito_EEE@Kirstens-MacBook-Pro:~/virtualenvs/EEE_198$ docker start larq_xcomp
larq_xcomp
kirstenhipolito_EEE@Kirstens-MacBook-Pro:~/virtualenvs/EEE_198$ docker ps
CONTAINER ID        IMAGE                              COMMAND             CREATED             STATUS              PORTS               NAMES
46717ebbb358        docker_lce_cross_compiler:latest   "/bin/bash"         4 minutes ago       Up 2 minutes                            larq_xcomp
kirstenhipolito_EEE@Kirstens-MacBook-Pro:~/virtualenvs/EEE_198$ docker exec -it larq_xcomp /bin/bash
root@46717ebbb358:/tmp/lce-volume# ls
compute-engine
root@46717ebbb358:/tmp/lce-volume# cd compute-engine/
root@46717ebbb358:/tmp/lce-volume/compute-engine# ls
BUILD               LICENSE      WORKSPACE             bazel-out         configure.sh  larq_compute_engine
CODE_OF_CONDUCT.md  MANIFEST.in  bazel-bin             bazel-testlogs    docs          setup.py
CONTRIBUTING.md     README.md    bazel-compute-engine  build_pip_pkg.sh  examples      third_party
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build \
>     --config=aarch64 \
>     //examples:lce_minimal
Starting local Bazel server and connecting to it...
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (63 packages loaded, 8418 targets configured).
INFO: Found 1 target...
INFO: Deleting stale sandbox base /root/.cache/bazel/_bazel_root/dffd717f69f2a652f7282bdcc53b6783/sandbox
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 15.291s, Critical Path: 0.20s
INFO: 0 processes.
INFO: Build completed successfully, 1 total action
root@46717ebbb358:/tmp/lce-volume/compute-engine# cd examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine/examples# ls
BUILD  converter_examples.py  lce_minimal.cc
root@46717ebbb358:/tmp/lce-volume/compute-engine/examples# cd ..
root@46717ebbb358:/tmp/lce-volume/compute-engine# nano examples/lce_minimal.cc 
bash: nano: command not found
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 2.614s, Critical Path: 2.18s
INFO: 2 processes: 2 local.
INFO: Build completed successfully, 3 total actions
root@46717ebbb358:/tmp/lce-volume/compute-engine# cd bazel-
bash: cd: bazel-: No such file or directory
root@46717ebbb358:/tmp/lce-volume/compute-engine# cd bazel-bin
root@46717ebbb358:/tmp/lce-volume/compute-engine/bazel-bin# ls
examples  external  larq_compute_engine
root@46717ebbb358:/tmp/lce-volume/compute-engine/bazel-bin# cd ex
examples/ external/ 
root@46717ebbb358:/tmp/lce-volume/compute-engine/bazel-bin# cd examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine/bazel-bin/examples# ls
_objs  lce_minimal  lce_minimal-2.params  lce_minimal.runfiles  lce_minimal.runfiles_manifest
root@46717ebbb358:/tmp/lce-volume/compute-engine/bazel-bin/examples# cd ../../../..
root@46717ebbb358:/tmp# la
ccM7e08t  hsperfdata_root  lce-volume
root@46717ebbb358:/tmp# cd lce-volume/compute-engine/
root@46717ebbb358:/tmp/lce-volume/compute-engine# ls bazel-bin
examples  external  larq_compute_engine
root@46717ebbb358:/tmp/lce-volume/compute-engine# ls
BUILD               LICENSE      WORKSPACE             bazel-out         configure.sh  larq_compute_engine
CODE_OF_CONDUCT.md  MANIFEST.in  bazel-bin             bazel-testlogs    docs          setup.py
CONTRIBUTING.md     README.md    bazel-compute-engine  build_pip_pkg.sh  examples      third_party
root@46717ebbb358:/tmp/lce-volume/compute-engine# ls bazel-bin/external/
FP16   XNNPACK  com_google_absl  farmhash_archive  flatbuffers     psimd
FXdiv  clog     cpuinfo          fft2d             org_tensorflow  pthreadpool
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# ls bazel-bin
external  larq_compute_engine
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 2.302s, Critical Path: 1.91s
INFO: 2 processes: 2 local.
INFO: Build completed successfully, 6 total actions
root@46717ebbb358:/tmp/lce-volume/compute-engine# ls bazel-bin
examples  external  larq_compute_engine
root@46717ebbb358:/tmp/lce-volume/compute-engine# ls bazel-bin/examples/
_objs  lce_minimal  lce_minimal-2.params  lce_minimal.runfiles  lce_minimal.runfiles_manifest
root@46717ebbb358:/tmp/lce-volume/compute-engine# cd bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine/bazel-bin/examples# nano examples/lce_minimal.cc 
bash: nano: command not found
root@46717ebbb358:/tmp/lce-volume/compute-engine/bazel-bin/examples# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine/bazel-bin/examples# cd ../..
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 2.361s, Critical Path: 2.01s
INFO: 2 processes: 2 local.
INFO: Build completed successfully, 6 total actions
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 2.337s, Critical Path: 1.99s
INFO: 2 processes: 2 local.
INFO: Build completed successfully, 6 total actions
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 2.341s, Critical Path: 2.00s
INFO: 2 processes: 2 local.
INFO: Build completed successfully, 6 total actions
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
ERROR: /tmp/lce-volume/compute-engine/examples/BUILD:8:1: C++ compilation of rule '//examples:lce_minimal' failed (Exit 1)
examples/lce_minimal.cc: In function 'int main(int, char**)':
examples/lce_minimal.cc:47:55: error: expected ',' or ';' before '(' token
   47 |   tflite::InterpreterBuilder builder(*model, resolver)(&interpreter,num_threads);
      |                                                       ^
Target //examples:lce_minimal failed to build
INFO: Elapsed time: 1.976s, Critical Path: 1.68s
INFO: 0 processes.
FAILED: Build did NOT complete successfully
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 2.232s, Critical Path: 1.93s
INFO: 2 processes: 2 local.
INFO: Build completed successfully, 6 total actions
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
ERROR: /tmp/lce-volume/compute-engine/examples/BUILD:8:1: C++ compilation of rule '//examples:lce_minimal' failed (Exit 1)
examples/lce_minimal.cc: In function 'int main(int, char**)':
examples/lce_minimal.cc:60:20: error: 'chrono' has not been declared
   60 |       auto start = chrono::steady_clock::now();
      |                    ^~~~~~
examples/lce_minimal.cc:63:18: error: 'chrono' has not been declared
   63 |       auto end = chrono::steady_clock::now();
      |                  ^~~~~~
examples/lce_minimal.cc:65:51: error: 'chrono' has not been declared
   65 |       std::cout << "Time of invoke (ms/FPS): " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << std::endl;
      |                                                   ^~~~~~
examples/lce_minimal.cc:65:73: error: 'chrono' has not been declared
   65 |       std::cout << "Time of invoke (ms/FPS): " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << std::endl;
      |                                                                         ^~~~~~
examples/lce_minimal.cc:66:24: error: 'chrono' has not been declared
   66 |       ave_invoke_ms += chrono::duration_cast<chrono::milliseconds>(end - start).count();
      |                        ^~~~~~
examples/lce_minimal.cc:66:46: error: 'chrono' has not been declared
   66 |       ave_invoke_ms += chrono::duration_cast<chrono::milliseconds>(end - start).count();
      |                                              ^~~~~~
Target //examples:lce_minimal failed to build
INFO: Elapsed time: 1.992s, Critical Path: 1.70s
INFO: 0 processes.
FAILED: Build did NOT complete successfully
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
ERROR: /tmp/lce-volume/compute-engine/examples/BUILD:8:1: C++ compilation of rule '//examples:lce_minimal' failed (Exit 1)
examples/lce_minimal.cc: In function 'int main(int, char**)':
examples/lce_minimal.cc:60:25: error: 'std::hrono' has not been declared
   60 |       auto start = std::hrono::steady_clock::now();
      |                         ^~~~~
examples/lce_minimal.cc:65:74: error: 'chrono' was not declared in this scope; did you mean 'std::chrono'?
   65 |       std::cout << "Time of invoke (ms): " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << std::endl;
      |                                                                          ^~~~~~
      |                                                                          std::chrono
In file included from examples/lce_minimal.cc:4:
/root/.cache/bazel/_bazel_root/dffd717f69f2a652f7282bdcc53b6783/external/aarch64_compiler/aarch64-none-linux-gnu/include/c++/9.2.1/chrono:59:13: note: 'std::chrono' declared here
   59 |   namespace chrono
      |             ^~~~~~
examples/lce_minimal.cc:65:60: error: parse error in template argument list
   65 |       std::cout << "Time of invoke (ms): " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << std::endl;
      |                                                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
examples/lce_minimal.cc:66:37: error: parse error in template argument list
   66 |       ave_invoke_ms += std::chrono::duration_cast<chrono::milliseconds>(end - start).count();
      |                                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Target //examples:lce_minimal failed to build
INFO: Elapsed time: 2.055s, Critical Path: 1.73s
INFO: 0 processes.
FAILED: Build did NOT complete successfully
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 2.201s, Critical Path: 1.93s
INFO: 2 processes: 2 local.
INFO: Build completed successfully, 3 total actions
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 2.276s, Critical Path: 1.94s
INFO: 2 processes: 2 local.
INFO: Build completed successfully, 6 total actions
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 
root@46717ebbb358:/tmp/lce-volume/compute-engine# rm -rf bazel-bin/examples/
root@46717ebbb358:/tmp/lce-volume/compute-engine# bazel build     --config=aarch64     //examples:lce_minimal
WARNING: option '--crosstool_top' was expanded to from both option '--config=manylinux2010' (source /tmp/lce-volume/compute-engine/.bazelrc) and option '--config=aarch64' (source command line options)
WARNING: /tmp/lce-volume/compute-engine/larq_compute_engine/tflite/kernels/BUILD:50:1: in cc_library rule //larq_compute_engine/tflite/kernels:lce_op_kernels: Target '//larq_compute_engine/tflite/kernels:lce_op_kernels' violates visibility of target '@org_tensorflow//tensorflow/lite/c:c_api_internal'. Continuing because --nocheck_visibility is active
INFO: Analyzed target //examples:lce_minimal (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //examples:lce_minimal up-to-date:
  bazel-bin/examples/lce_minimal
INFO: Elapsed time: 2.302s, Critical Path: 2.01s
INFO: 2 processes: 2 local.
INFO: Build completed successfully, 6 total actions
root@46717ebbb358:/tmp/lce-volume/compute-engine# vim examples/lce_minimal.cc 

#include <cstdio>
#include <ctime>
#include <iostream>
#include <chrono>
#include <string>
#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This file is based on the TF lite minimal example where the
// "BuiltinObResolver" is modified to include the "Larq Compute Engine" custom
// ops. Here we read a binary model form disk and perform inference by using the
// C++ interface. See the BUILD file in this directory to see an example of
// linking "Larq Compute Engine" cutoms ops to your inference binary.

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "lce_minimal <tflite model>\n");
    return 1;
  }

  int num_runs = 50;
  //clock_t ave_invoke_ms = 0;
  float ave_invoke_ms = 0;
  clock_t time_req_1;
  int num_threads = std::atoi(argv[2]);

  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  std::unique_ptr<Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter,num_threads);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Initial invoke
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  std::cout << "Testing invoke() on " << argv[2] << "threads for " << num_runs << "times." << std::endl;

  // Run multiple iterations of invoke
  for (int i = 0; i < num_runs; i++) {
      auto start = std::chrono::steady_clock::now();
      // Run inference
      TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
      auto end = std::chrono::steady_clock::now();

      std::cout << "Time of invoke (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
      ave_invoke_ms += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      // ave_invoke_ms += time_req_1;
  }
"examples/lce_minimal.cc" 76L, 2816C                                                                                                            
