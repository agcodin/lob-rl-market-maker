import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# -march=native is opt-in: it breaks universal2 (multi-arch) builds, which is
# the default on macOS when ARCHFLAGS is unset.
flags = ["-O3", "-fno-omit-frame-pointer"]
if os.environ.get("LOBRL_NATIVE") == "1":
    flags.append("-mcpu=native" if os.uname().machine == "arm64" else "-march=native")

ext = Pybind11Extension(
    "lobrl._lobcore",
    ["cpp/src/bindings.cpp"],
    include_dirs=["cpp/include"],
    cxx_std=20,
    extra_compile_args=flags,
)

setup(
    name="lobrl",
    version="0.1.0",
    description="Low-latency limit order book + PPO market maker",
    packages=["lobrl"],
    package_dir={"": "python"},
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
)
