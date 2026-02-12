# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import fcntl
import multiprocessing
import atexit
import os
import signal
# import sys
import shlex
import os.path
import pathlib
import select
import shutil
import logging
import subprocess
import tarfile
import tempfile
import time
# import re

import warnings

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import distutils.util

from tvm.micro.project_api import server

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.WARNING)

DBG = False
# DBG = True

PRINT = False
# PRINT = True

PROJECT_DIR = pathlib.Path(os.path.dirname(__file__) or os.getcwd())


MODEL_LIBRARY_FORMAT_RELPATH = "model.tar"


IS_TEMPLATE = not os.path.exists(os.path.join(PROJECT_DIR, MODEL_LIBRARY_FORMAT_RELPATH))

# Used this size to pass most CRT tests in TVM.
# WORKSPACE_SIZE_BYTES = 4 * 1024 * 1024
WORKSPACE_SIZE_BYTES = 2 * 1024 * 1024
# WORKSPACE_SIZE_BYTES = 1 * 1024 * 1024

CPU_FREQ = 100e6

CMAKEFILE_FILENAME = "CMakeLists.txt"

# The build target given to make
BUILD_TARGET = "build/main"

ARCH = "rv32gc"
ABI = "ilp32d"
TRIPLE = "riscv32-unknown-elf"
TOOLCHAIN = "gcc"
NPROC = multiprocessing.cpu_count()


def str2bool(value, allow_none=False):
    if value is None:
        assert allow_none, "str2bool received None value while allow_none=False"
        return value
    return bool(value) if isinstance(value, (int, bool)) else bool(distutils.util.strtobool(value))


def check_call(cmd_args, *args, **kwargs):
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    _LOG.info("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args))
    return subprocess.check_call(cmd_args, *args, **kwargs)


class Handler(server.ProjectAPIHandler):
    BUILD_TARGET = "build/main"

    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None
        if DBG:
            self.elfdest = tempfile.mkstemp(dir="/tmp/elfs")[1]
            self.outputs = b""

    def server_info_query(self, tvm_version):
        return server.ServerInfo(
            platform_name="host",
            is_template=IS_TEMPLATE,
            model_library_format_path=""
            if IS_TEMPLATE
            else PROJECT_DIR / MODEL_LIBRARY_FORMAT_RELPATH,
            project_options=[
                server.ProjectOption(
                    "verbose",
                    optional=["build"],
                    type="bool",
                    default=False,
                    help="Run make with verbose output",
                ),
                server.ProjectOption(
                    "quiet",
                    optional=["build"],
                    type="bool",
                    default=True,
                    help="Supress all compilation messages",
                ),
                server.ProjectOption(
                    "debug",
                    optional=["build"],
                    type="bool",
                    default=False,
                    help="Build with debugging symbols and -O0",
                ),
                server.ProjectOption(
                    "workspace_size_bytes",
                    optional=["generate_project"],
                    type="int",
                    default=WORKSPACE_SIZE_BYTES,
                    help="Sets the value of TVM_WORKSPACE_SIZE_BYTES.",
                ),
                server.ProjectOption(
                    "arch",
                    optional=["build"],
                    default=ARCH,
                    type="str",
                    help="Name used ARCH.",
                ),
                server.ProjectOption(
                    "abi",
                    optional=["build"],
                    default=ABI,
                    type="str",
                    help="Name used ABI.",
                ),
                server.ProjectOption(
                    "cpu_arch",
                    optional=["generate_project"],
                    # default=None,
                    type="str",
                    help="Name used CPU_ARCH.",
                ),
                server.ProjectOption(
                    "toolchain",
                    optional=["build"],
                    default=TOOLCHAIN,
                    choices=["gcc", "llvm"],
                    type="str",
                    help="Name used TOOLCHAIN.",
                ),
                server.ProjectOption(
                    "llvm_dir",
                    optional=["build"],
                    default=None,
                    type="str",
                    help="Path to LLVM install directory",
                ),
                server.ProjectOption(
                    "gcc_prefix",
                    optional=["build"],
                    default="",
                    type="str",
                    help="Name used COMPILER.",
                ),
                server.ProjectOption(
                    "gcc_name",
                    optional=["build"],
                    default=TRIPLE,
                    type="str",
                    help="Name used COMPILER.",
                ),
                server.ProjectOption(
                    "core",
                    optional=["generate_project" "build", "flash", "open_transport"],
                    default="cv32e40x",
                    type="str",
                    help="Used base core.",
                ),
                server.ProjectOption(
                    "vproc_pipelines",
                    optional=["generate_project" "build", "flash", "open_transport"],
                    default=None,
                    type="str",
                    help="Used base core.",
                ),
                server.ProjectOption(
                    "mem_size",
                    optional=["generate_project" "build", "flash", "open_transport"],
                    default=4194304,
                    type="int",
                    help="Memory size.",
                ),
                server.ProjectOption(
                    "mem_width",
                    optional=["generate_project" "build", "flash", "open_transport"],
                    default=32,
                    type="int",
                    help="Memory port width.",
                ),
                server.ProjectOption(
                    "vreg_width",
                    optional=["generate_project" "build", "flash", "open_transport"],
                    default=128,
                    type="int",
                    help="Used VLEN.",
                ),
                server.ProjectOption(
                    "vlane_width",
                    optional=["generate_project" "build", "flash", "open_transport"],
                    default=64,
                    type="int",
                    help="Used VLANE_W.",
                ),
                server.ProjectOption(
                    "vmem_width",
                    optional=["generate_project" "build", "flash", "open_transport"],
                    default=32,
                    type="int",
                    help="Vector memory port width.",
                ),
                server.ProjectOption(
                    "vicuna2_model_dir",
                    optional=["generate_project" "build", "flash", "open_transport"],
                    default=None,
                    type="str",
                    help="Path to build_model.",
                ),
                server.ProjectOption(
                    "vicuna2_bsp_dir",
                    optional=["generate_project" "build", "flash", "open_transport"],
                    default=None,
                    type="str",
                    help="Path to vicuna2 bsp.",
                ),
                server.ProjectOption(
                    "cpu_freq",
                    optional=["generate_project", "build"],  # TODO: check
                    type="int",
                    default=CPU_FREQ,
                    help="Sets the value of VICUNA_CPU_FREQ_HZ.",
                ),
                server.ProjectOption(
                    "trace",
                    optional=["flash", "open_transport"],
                    type="bool",
                    default=False,
                    help="Write instruction trace to file",
                ),
                server.ProjectOption(
                    "full_trace",
                    optional=["flash", "open_transport"],
                    type="bool",
                    default=False,
                    help="Write vcd trace to file",
                ),
                server.ProjectOption(
                    "verilator_install_dir",
                    required=["flash"],
                    type="str",
                    help="Path to verilator installation.",
                ),
            ],
        )

    # These files and directories will be recursively copied into generated projects from the CRT.
    CRT_COPY_ITEMS = ("include", "CMakeLists.txt", "src")

    def _populate_cmake(
        self,
        cmakefile_template_path: pathlib.Path,
        cmakefile_path: pathlib.Path,
        memory_size: int,
        verbose: bool,
    ):
        """Generate CMakeList file from template."""

        with open(cmakefile_path, "w") as cmakefile_f:
            with open(cmakefile_template_path, "r") as cmakefile_template_f:
                for line in cmakefile_template_f:
                    cmakefile_f.write(line)
                cmakefile_f.write(
                    f"target_compile_definitions(main PUBLIC -DTVM_WORKSPACE_SIZE_BYTES={memory_size})\n"
                )
                if verbose:
                    cmakefile_f.write(f"set(CMAKE_VERBOSE_MAKEFILE TRUE)\n")

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        # Make project directory.
        project_dir.mkdir(parents=True)
        current_dir = pathlib.Path(__file__).parent.absolute()

        # Copy ourselves to the generated project. TVM may perform further build steps on the generated project
        # by launching the copy.
        shutil.copy2(__file__, project_dir / os.path.basename(__file__))

        # Place Model Library Format tarball in the special location, which this script uses to decide
        # whether it's being invoked in a template or generated project.
        project_model_library_format_path = project_dir / MODEL_LIBRARY_FORMAT_RELPATH
        shutil.copy2(model_library_format_path, project_model_library_format_path)

        # Extract Model Library Format tarball.into <project_dir>/model.
        extract_path = project_dir / project_model_library_format_path.stem
        with tarfile.TarFile(project_model_library_format_path) as tf:
            os.makedirs(extract_path)
            tf.extractall(path=extract_path)

        # Populate CRT.
        crt_path = project_dir / "crt"
        os.mkdir(crt_path)
        for item in self.CRT_COPY_ITEMS:
            src_path = standalone_crt_dir / item
            dst_path = crt_path / item
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Populate CMake file
        self._populate_cmake(
            current_dir / f"{CMAKEFILE_FILENAME}.template",
            project_dir / CMAKEFILE_FILENAME,
            options.get("workspace_size_bytes", WORKSPACE_SIZE_BYTES),
            str2bool(options.get("verbose"), False),
        )
        cmake_path = project_dir / "cmake"
        os.mkdir(cmake_path)
        shutil.copytree(current_dir / "cmake", cmake_path, dirs_exist_ok=True)

        # Populate crt-config.h
        crt_config_dir = project_dir / "crt_config"
        crt_config_dir.mkdir()
        shutil.copy2(
            current_dir / "crt_config" / "crt_config.h",
            crt_config_dir / "crt_config.h",
        )

        # Populate src/
        src_dir = project_dir / "src"
        src_dir.mkdir()
        shutil.copy2(
            current_dir / "src" / "main.cc",
            src_dir / "main.cc",
        )
        shutil.copy2(
            current_dir / "src" / "platform.cc",
            src_dir / "platform.cc",
        )

    def build(self, options):
        if PRINT:
            print("build")
        build_dir = PROJECT_DIR / "build"
        build_dir.mkdir()
        cmake_args = []
        debug = options.get("debug", False)
        build_type = "Debug" if debug else "Release"
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
        cpu_freq = options.get("cpu_freq", CPU_FREQ)
        cmake_args.append(f"-DVICUNA_CPU_FREQ_HZ={cpu_freq}")
        cmake_args.append("-DTOOLCHAIN=" + options.get("toolchain", TOOLCHAIN))
        llvm_dir = options.get("llvm_dir", None)
        if llvm_dir:
            cmake_args.append("-DLLVM_DIR=" + llvm_dir)
        vicuna2_bsp_dir = options.get("vicuna2_bsp_dir", None)
        if vicuna2_bsp_dir:
            cmake_args.append("-DVICUNA2_BSP_DIR=" + vicuna2_bsp_dir)
        cmake_args.append("-DRISCV_ARCH=" + options.get("arch", ARCH))
        cmake_args.append("-DRISCV_ABI=" + options.get("abi", ABI))
        cmake_args.append("-DRISCV_ABI=" + options.get("abi", ABI))
        cmake_args.append("-DRISCV_ELF_GCC_PREFIX=" + options.get("gcc_prefix", ""))
        cmake_args.append("-DRISCV_ELF_GCC_BASENAME=" + options.get("gcc_name", TRIPLE))
        # print("cmake_args", cmake_args)
        if str2bool(options.get("quiet"), True):
            check_call(["cmake", "..", *cmake_args], cwd=build_dir, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            check_call(["make", f"-j{NPROC}"], cwd=build_dir, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            check_call(["cmake", "..", *cmake_args], cwd=build_dir)
            check_call(["make", f"-j{NPROC}"], cwd=build_dir)

    def get_model_cmake_args(self, options):
        ret = []

        def filter_riscv_arch(arch):
            ret = arch.replace("_zvl128b", "")
            ret = ret.replace("_zicsr", "")
            ret = ret.replace("_zifencei", "")
            return ret

        arch = options.get("arch", ARCH)
        vlen = options.get("vreg_width", 128)
        riscv_arch = filter_riscv_arch(arch)
        ret.append(f"-DRISCV_ARCH={riscv_arch}")
        ret.append(f"-DVREG_W={vlen}")
        if options.get("trace", False):
            ret.append("-DTRACE=ON")
        if options.get("trace_full", False):
            ret.append("-DTRACE_FULL=ON")
        vmem_width = options.get("vmem_width", 32)
        mem_width = options.get("mem_width", 32)
        vlane_width = options.get("vlane_width", 64)
        core = options.get("core", "cv32e40x")
        mem_size = options.get("mem_size", 4194304)
        if vmem_width is not None:
            ret.append(f"-DVMEM_W={vmem_width}")
        if mem_width is not None:
            ret.append(f"-DMEM_W={mem_width}")
        if vlane_width is not None:
            ret.append(f"-DVLANE_W={vlane_width}")
        if mem_size is not None:
            # TODO: use in cmake!
            ret.append(f"-DMEM_SZ={mem_size}")
        ret.append(f"-DSCALAR_CORE={core}")

        return ret

    def prepare_environment(self, env: dict, options):
        new_path = env.get("PATH", "")
        gcc_prefix = options.get("gcc_prefix", None)
        if gcc_prefix is not None:
            new_path = f"{gcc_prefix}/bin:{new_path}"
        verilator_install_dir = options.get("verilator_install_dir", None)
        if verilator_install_dir is not None:
            new_path = f"{verilator_install_dir}/bin:{new_path}"
        env["PATH"] = new_path
        return env

    def flash(self, options):
        if PRINT:
            print("flash")
        model_build_dir = PROJECT_DIR / "build_model"
        model_build_dir.mkdir(exist_ok=True)
        env = self.prepare_environment(os.environ.copy(), options)
        vicuna2_model_dir = options.get("vicuna2_model_dir", None)
        cmake_configure_args = ["-S", str(vicuna2_model_dir), "-B", str(model_build_dir), *self.get_model_cmake_args(options)]
        cmake_build_args = ["--build", str(model_build_dir), f"-j{NPROC}"]
        if str2bool(options.get("quiet"), True):
            check_call(["cmake", *cmake_configure_args], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, env=env)
            check_call(["cmake", *cmake_build_args], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, env=env)
        else:
            check_call(["cmake", *cmake_configure_args], env=env)
            check_call(["cmake", *cmake_build_args], env=env)

    def _set_nonblock(self, fd):
        flag = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
        new_flag = fcntl.fcntl(fd, fcntl.F_GETFL)
        assert (new_flag & os.O_NONBLOCK) != 0, "Cannot set file descriptor {fd} to non-blocking"

    def open_transport(self, options):
        if PRINT:
            print("open_transport")
        # self._proc = subprocess.Popen(
        #     [self.BUILD_TARGET], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        # )
        # print("PROJECT_DIR", PROJECT_DIR)
        mem_width = options.get("mem_width", 32)
        mem_size = options.get("mem_size", 4194304)
        mem_latency = 1
        extra_cycles = 1
        trace = options.get("trace", False)
        trace_full = options.get("trace_full", False)
        if trace or trace_full:
            raise NotImplementedError
        model_build_dir = PROJECT_DIR / "build_model"
        vicuna_exe = model_build_dir / "verilated_model"
        assert vicuna_exe.is_file()
        build_dir = PROJECT_DIR / "build"
        path_file = build_dir / "main.path"
        assert path_file.is_file()
        args = [
            str(vicuna_exe),
            str(path_file),
            str(mem_width),
            str(mem_size),
            str(mem_latency),
            str(extra_cycles),
            # "instr_trace.log",
        ]
        # print("CWD", os.getcwd())
        # input(">")
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            preexec_fn=os.setsid,
            cwd=PROJECT_DIR
        )
        # print("A")
        self._set_nonblock(self._proc.stdin.fileno())
        self._set_nonblock(self._proc.stdout.fileno())
        self._drain_until_rpc_start()
        # while True:
        #     self.read_transport(1000, 10.0)
        #     time.sleep(1)
        # input("?")
        atexit.register(lambda: self.close_transport())
        return server.TransportTimeouts(
            session_start_retry_timeout_sec=0,
            session_start_timeout_sec=0,
            session_established_timeout_sec=0,
        )

    def close_transport(self):
        if PRINT:
            print("close_transport")
        if DBG:
            outfile = str(self.elfdest) + ".out"
            with open(outfile, "wb") as f:
                f.write(self.outputs)
        if self._proc is not None:
            proc = self._proc
            # pgrp = os.getpgid(proc.pid)
            self._proc = None
            proc.terminate()
            proc.kill()
            proc.wait()
            # os.killpg(pgrp, signal.SIGKILL)

    def _await_ready(self, rlist, wlist, timeout_sec=None, end_time=None):
        if timeout_sec is None and end_time is not None:
            timeout_sec = max(0, end_time - time.monotonic())

        rlist, wlist, xlist = select.select(rlist, wlist, rlist + wlist, timeout_sec)
        if not rlist and not wlist and not xlist:
            raise server.IoTimeoutError()

        return True

    def _drain_until_rpc_start(self, timeout=100.0):
        if PRINT:
            print("_drain_until_rpc_start")
        end = time.time() + timeout
        hist = b""
        while time.time() < end:
            fd = self._proc.stdout.fileno()
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                continue

            b = os.read(fd, 1)
            hist += b
            if DBG:
                self.outputs += b
            if not b:
                continue

            if b == b'\xfe':
                # push back into buffer
                self._rx_buffer = b
                return

        if DBG:
            outfile = str(self.elfdest) + ".out"
            # self.outputs += hist
            with open(outfile, "wb") as f:
                f.write(self.outputs)
        raise RuntimeError("RPC start byte not found")

    def read_transport(self, n, timeout_sec):
        if PRINT:
            print("read_transport", n)
        if self._rx_buffer:
            data = self._rx_buffer
            self._rx_buffer = b""
            if PRINT:
                print("ret", data)
            return data

        if self._proc is None:
            raise server.TransportClosedError()

        fd = self._proc.stdout.fileno()
        end_time = None if timeout_sec is None else time.monotonic() + timeout_sec

        try:
            self._await_ready([fd], [], end_time=end_time)
            to_return = os.read(fd, n)
        except BrokenPipeError:
            to_return = 0

        if not to_return:
            self.close_transport()
            raise server.TransportClosedError()
        if PRINT:
            print("ret", to_return)
        if DBG:
            self.outputs += to_return

        return to_return

    def write_transport(self, data, timeout_sec):
        if PRINT:
            print("write_transport", data)
        if self._proc is None:
            raise server.TransportClosedError()

        fd = self._proc.stdin.fileno()
        end_time = None if timeout_sec is None else time.monotonic() + timeout_sec

        data_len = len(data)
        while data:
            self._await_ready([], [fd], end_time=end_time)
            try:
                num_written = os.write(fd, data)
            except BrokenPipeError:
                num_written = 0

            if not num_written:
                self.disconnect_transport()
                raise server.TransportClosedError()

            data = data[num_written:]


if __name__ == "__main__":
    server.main(Handler())
