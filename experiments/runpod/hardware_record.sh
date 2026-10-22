#!/bin/sh
# Record the hardware and software of a Runpod Pod before an experiment runs (D-56).
# Usage, inside the Pod: sh hardware_record.sh <results dir>
# It writes <results dir>/hardware/ and never fails, so a missing tool does not stop the experiment.
# It reads only an allowlist of RUNPOD_* variables, because the Pod environment can hold an API key.
set -u

OUT=${1:-/results}/hardware
mkdir -p "$OUT"

cap() {  # $1 file name, then the command; keeps stderr, never fails
    f=$1; shift
    "$@" > "$OUT/$f" 2>&1 || echo "exit $?" >> "$OUT/$f"
}

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/recorded_at.txt"

# Pod identity. Do not dump the full environment.
for v in RUNPOD_POD_ID RUNPOD_POD_HOSTNAME RUNPOD_DC_ID RUNPOD_GPU_COUNT RUNPOD_CPU_COUNT \
         RUNPOD_MEM_GB CUDA_VERSION NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES; do
    eval "val=\${$v:-}"
    printf '%s=%s\n' "$v" "$val"
done > "$OUT/pod_env.txt"

# GPU.
cap nvidia-smi-q.txt nvidia-smi -q
cap nvidia-smi-topo.txt nvidia-smi topo -m
cap nvidia-smi-nvlink.txt nvidia-smi nvlink -s
cap gpu.csv nvidia-smi --format=csv --query-gpu=index,name,uuid,driver_version,vbios_version,pci.bus_id,\
memory.total,power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.memory,\
clocks.applications.graphics,ecc.mode.current,mig.mode.current,compute_mode,persistence_mode,\
pcie.link.gen.current,pcie.link.width.current,temperature.gpu
cap nvcc.txt nvcc --version

# Python stack.
cap python.txt python3 -c "
import platform, sys
print('python', sys.version.split()[0], platform.machine())
try:
    import torch
    print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, 'sm_%d%d' % (p.major, p.minor), p.multi_processor_count, 'SMs', p.total_memory // 2**20, 'MiB')
except Exception as e:
    print('torch:', e)
try:
    import vllm
    print('vllm', vllm.__version__)
except Exception as e:
    print('vllm:', e)
"
cap pip-freeze.txt python3 -m pip freeze

# CPU, memory, disk, OS and container limits.
cap lscpu.txt lscpu
cap nproc.txt nproc
cap meminfo.txt head -5 /proc/meminfo
cap df.txt df -h
cap uname.txt uname -a
cap os-release.txt cat /etc/os-release
cap cgroup.txt sh -c 'for f in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max; do echo "$f: $(cat $f 2>&1)"; done'

echo "hardware record written to $OUT"
