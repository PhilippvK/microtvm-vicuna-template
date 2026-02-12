/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file main.cc
 * \brief main entry point for host subprocess-based CRT
 */

extern "C" void *__dso_handle = 0;

#include <inttypes.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/aot_executor_module.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <unistd.h>

#include <iostream>

#include "crt_config.h"
extern "C" {
#include "uart.h"
}

#ifdef TVM_HOST_USE_GRAPH_EXECUTOR_MODULE
#include <tvm/runtime/crt/graph_executor_module.h>
#endif

extern "C" {

ssize_t MicroTVMWriteFunc(void* context, const uint8_t* data, size_t num_bytes) {
  uart_write(num_bytes, (const char*)data);
  return num_bytes;
}

}

static char** g_argv = NULL;

/*int testonly_reset_server(TVMValue* args, int* type_codes, int num_args, TVMValue* out_ret_value,
                          int* out_ret_tcode, void* resource_handle) {
  execvp(g_argv[0], g_argv);
  perror("microTVM runtime: error restarting");
  return -1;
}*/

int main(int argc, char** argv) {
  // uart_printf("Hello World!\n");
  g_argv = argv;
  TVMPlatformInitialize();
  // uart_printf("platform initialized\n");
  microtvm_rpc_server_t rpc_server = MicroTVMRpcServerInit(&MicroTVMWriteFunc, nullptr);
  // uart_printf("server initialized\n");

#ifdef TVM_HOST_USE_GRAPH_EXECUTOR_MODULE
  CHECK_EQ(TVMGraphExecutorModule_Register(), kTvmErrorNoError,
           "failed to register GraphExecutor TVMModule");
#endif
  // uart_printf("executor initialized\n");

  // int error = TVMFuncRegisterGlobal("tvm.testing.reset_server",
  //                                   (TVMFunctionHandle)&testonly_reset_server, 0);
  // if (error) {
  //   fprintf(
  //       stderr,
  //       "microTVM runtime: internal error (error#: %x) registering global packedfunc; exiting\n",
  //       error);
  //   return 2;
  // }

  // setbuf(stdin, NULL);
  // setbuf(stdout, NULL);
  TVMLogf("microTVM Vicuna runtime - running");
  // uart_printf("log done\n");

  for (;;) {
    // uart_printf("loop\n");
    char buf[1];
    uart_read(1, buf);
    uint8_t c = buf[0];
    // printf("c=%c\n", c);
    // uart_printf("c=%c\n", c);
    // continue;
    uint8_t* cursor = &c;
    size_t bytes_to_process = 1;
    while (bytes_to_process > 0) {
      tvm_crt_error_t err = MicroTVMRpcServerLoop(rpc_server, &cursor, &bytes_to_process);
      if (err == kTvmErrorPlatformShutdown) {
        break;
      } else if (err != kTvmErrorNoError) {
        char buf[1024];
        //snprintf(buf, sizeof(buf), "microTVM runtime: MicroTVMRpcServerLoop error: %08x", err);
        TVMLogf("?Ret2?\n");
        // perror(buf);
        return 2;
      }
    }
  }
  TVMLogf("?Done?\n");
  return 0;
}
