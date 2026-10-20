#!/bin/sh
# E-30 preflight on a GH200 (amendment A1 after R-34): can we read and set the power knobs?
# The only writes are idempotent: each knob is set to its current value. Writes preflight.txt and
# prints a short verdict.
set -u
OUT=${1:-preflight}
mkdir -p "$OUT"
{
    echo "== root"; sudo -n true && echo "sudo: yes" || echo "sudo: no"
    echo "== cpu"; nproc; lscpu | grep -E "Model name|Socket|NUMA node" || true
    echo "== nvidia-smi power"; nvidia-smi -q -d POWER
    echo "== hwmon (both layouts)"
    for f in /sys/class/hwmon/hwmon*/power*_cap /sys/class/hwmon/hwmon*/device/power*_cap; do
        [ -e "$f" ] && printf '%s = %s\n' "$f" "$(cat "$f" 2>/dev/null)"
    done
    echo "== stress-ng"; command -v stress-ng || echo "stress-ng missing (sudo apt-get install -y stress-ng)"
} > "$OUT/preflight.txt" 2>&1

verdict() { printf '%s\n' "$*" | tee -a "$OUT/verdict.txt"; }
: > "$OUT/verdict.txt"

sudo -n true 2>/dev/null && verdict "root: yes" || { verdict "root: NO (stop: caps cannot be set)"; exit 1; }
[ "$(nproc)" -ge 62 ] && verdict "cores: $(nproc)" || verdict "cores: $(nproc) (fewer than 62: sweep.py will refuse to run)"

module=$(nvidia-smi -q -d POWER | sed -n '/Module Power Readings/,$p')
if [ -z "$module" ]; then
    verdict "module power section: ABSENT (KILL 1 at the GPU scope only; report it)"
else
    cur=$(printf '%s\n' "$module" | sed -n 's/^ *Current Power Limit *: *\([0-9.]*\) W.*/\1/p' | head -1)
    min=$(printf '%s\n' "$module" | sed -n 's/^ *Min Power Limit *: *\([0-9.]*\) W.*/\1/p' | head -1)
    max=$(printf '%s\n' "$module" | sed -n 's/^ *Max Power Limit *: *\([0-9.]*\) W.*/\1/p' | head -1)
    verdict "module limit: current=${cur} min=${min} max=${max}"
    if [ -n "$cur" ] && sudo nvidia-smi -pl "${cur%.*}" -sc 1 >> "$OUT/preflight.txt" 2>&1; then
        verdict "module limit writable: yes (set to its current value)"
    else
        verdict "module limit writable: NO (only the default limit can be tested)"
    fi
    if [ -n "$min" ] && [ -n "$max" ]; then
        awk -v lo="$min" -v hi="$max" 'BEGIN { exit !(680 >= lo && 680 <= hi) }' \
            && verdict "680 W within [min, max]: yes" || verdict "680 W within [min, max]: NO (drop the 680 W limit)"
    fi
fi

caps=0
for f in /sys/class/hwmon/hwmon*/power*_cap /sys/class/hwmon/hwmon*/device/power*_cap; do
    [ -e "$f" ] || continue
    caps=1
    v=$(cat "$f" 2>/dev/null)
    if [ -n "$v" ] && sudo sh -c "echo $v > $f" 2>/dev/null; then
        verdict "CPU cap writable: $f (current $v uW)"
    else
        verdict "CPU cap NOT writable: $f (needs acpi_power_meter.force_cap_on=y on the kernel command line)"
    fi
done
[ "$caps" = 1 ] || verdict "CPU power caps: ABSENT (KILL 2 cannot be tested)"
command -v stress-ng > /dev/null && verdict "stress-ng: present" || verdict "stress-ng: MISSING"
