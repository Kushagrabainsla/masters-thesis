#!/bin/bash
# E-55 create-to-copy in one call (README Section 9; the E-54 script `e54_run.sh` with the E-55 paths and no capture
# folder): register the Pod, make the design manifest (make_manifest.py at the design commit), wait for the SSH port
# until 15 min after the create (then exit 3: the main session deletes the Pod), upload the package, the frozen scales
# file and the manifest, check the GPU name (abort rule 1) and the code hashes on the Pod against the manifest, start
# session_e55.sh with AUTO_STOP (once: a lock folder stops a second start), print each new line of session.log and the
# arms done, and when the session is done copy RES inside the AUTO_STOP window and compare the checksums. With
# WAITSTOP=1 it then waits until AUTO_STOP stops the Pod (the rehearsal check). It starts no Pod and deletes no Pod.
# Every failed step before the watch exits at once (L-24), so the main session gets the notice and deletes the Pod.
# Usage: e55_run.sh <pod_id> <pod_name> <ssh alias> <local results dir> "<session env>" "<registry note>"
#        <create epoch> <max watch s> "<GPU name part>" [WAITSTOP]
set -u
ID=$1; NAME=$2; A=$3; LOC=$4; SENV=$5; NOTE=$6; CREATED=$7; MAX=$8; GPU=$9; WAITSTOP=${10:-0}
SSH_WAIT=${SSH_WAIT:-900}          # seconds after the create without an SSH port before exit 3
B=/Users/kushagrabainsls/Desktop/mscs/masters-thesis; E=$B/experiments; R=$E/runpod
SP=/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/20255ed9-038a-44e5-8a60-f7885a8c54ee/scratchpad
C=$SP/ssh_config_$A
DESIGN=92c1dc5                       # the A1 design commit (README Section 12)
REM=/workspace/results/e55
mkdir -p "$LOC"
python3 $R/pods.py check "$ID" > /dev/null 2>&1 || python3 $R/pods.py add "$ID" "$NAME" E-55 pod --note "$NOTE"
MAN=$LOC/code_manifest.json
uv run --no-project --with numpy python -B $E/gemma4_kv_model/make_manifest.py --commit $DESIGN --out "$MAN" || \
    { echo "NO MANIFEST: DELETE THE POD NOW"; exit 5; }
pod_query() {  # $1 = the GraphQL fields of the pod; the read-only key from .env (never printed)
    python3 - "$B/.env" "$ID" "$1" <<'EOF'
import json, sys, urllib.request
key = next(l.split("=", 1)[1].strip() for l in open(sys.argv[1]) if l.startswith("RUNPOD_API_KEY="))
q = {"query": 'query { pod(input: {podId: "%s"}) { %s } }' % (sys.argv[2], sys.argv[3])}
r = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps(q).encode(), method="POST",
                           headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
print(json.dumps(json.load(urllib.request.urlopen(r, timeout=30))["data"]["pod"]))
EOF
}
python3 - "$B/.env" "$C" "$ID" "$A" "$CREATED" "$SP/known_hosts_$ID" "$SSH_WAIT" <<'EOF'
import json, os, sys, time, urllib.request
key = next(l.split("=", 1)[1].strip() for l in open(sys.argv[1]) if l.startswith("RUNPOD_API_KEY="))
q = {"query": 'query { pod(input: {podId: "%s"}) { desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } } } }' % sys.argv[3]}
created = int(sys.argv[5])
while time.time() < created + int(sys.argv[7]):
    try:
        r = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps(q).encode(), method="POST",
                                   headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        d = json.load(urllib.request.urlopen(r, timeout=30))["data"]["pod"]
    except Exception as e:  # a short API error must not end the wait
        print("query error:", type(e).__name__, flush=True); time.sleep(10); continue
    p = next((x for x in ((d.get("runtime") or {}).get("ports") or []) if x["privatePort"] == 22 and x["isIpPublic"]), None)
    if p:
        h = os.path.expanduser("~")
        # a known-hosts file per Pod: Runpod reuses an IP and port with a new host key
        open(sys.argv[2], "w").write(f"Host {sys.argv[4]}\n  HostName {p['ip']}\n  Port {p['publicPort']}\n  User root\n"
                                     f"  IdentityFile {h}/.ssh/runpod_ed25519\n  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n"
                                     f"  UserKnownHostsFile {sys.argv[6]}\n  ConnectTimeout 30\n  ConnectionAttempts 3\n"
                                     f"  ServerAliveInterval 30\n")
        print(f"ssh port {round(time.time() - created)} s after the create (uptime {d['runtime']['uptimeInSeconds']} s)", flush=True)
        break
    time.sleep(10)
else:
    print(f"NO SSH PORT {sys.argv[7]} s after the create: DELETE THE POD NOW", flush=True)
    sys.exit(3)
EOF
[ $? -eq 0 ] || exit 3
ok=0
for i in $(seq 1 30); do ssh -n -F $C $A true 2>/dev/null && { echo "sshd up after $i tries"; ok=1; break; }; perl -e 'select(undef,undef,undef,10)'; done
[ $ok = 1 ] || { echo "NO SSHD: DELETE THE POD NOW"; exit 4; }
try() {  # (E-54 incident, 2026-11-02) a command over a slow link: up to 5 tries, 15 s apart
    for k in 1 2 3 4 5; do "$@" && return 0; echo "try $k failed: $*" | cut -c1-160; perl -e 'select(undef,undef,undef,15)'; done
    return 1
}
die() { echo "RUN STOPPED: $1. DELETE THE POD NOW"; exit 5; }
P=/workspace/code
try ssh -n -F $C $A "mkdir -p $P/gemma4_kv_model $P/gemma4_kv_scaletie $P/runpod && test -d $P/gemma4_kv_model" || die "no $P"
ssh -n -F $C $A 'cat /workspace/boot_watchdog.log; tr "\0" "\n" < /proc/1/environ | grep -E "^(RUNPOD_POD_ID|RUNPOD_API_KEY)=" | cut -d= -f1 | sed "s/$/ is set/"'
try rsync -rlt --timeout=60 -e "ssh -F $C" --exclude /results/ --exclude /data/ --exclude __pycache__ \
    --exclude .pytest_cache $E/gemma4_kv_model/ $A:$P/gemma4_kv_model/ || die "the upload of gemma4_kv_model failed"
try rsync -rlt --timeout=60 -e "ssh -F $C" "$MAN" $A:$P/gemma4_kv_model/code_manifest.json || die "the upload of the manifest failed"
try rsync -rlt --timeout=60 -e "ssh -F $C" $E/gemma4_kv_scaletie/scales_e54.json $A:$P/gemma4_kv_scaletie/ || \
    die "the upload of the scales failed"
try rsync -rlt --timeout=60 -e "ssh -F $C" --exclude __pycache__ --exclude .pytest_cache --exclude pods.jsonl \
    --exclude HARDWARE_LOG.md $R/ $A:$P/runpod/ || die "the upload of runpod failed"
ssh -n -F $C $A "sha256sum $P/gemma4_kv_model/splice_e55.py $P/gemma4_kv_model/run_e55.py $P/gemma4_kv_model/analyze_e55.py $P/gemma4_kv_model/session_e55.sh $P/gemma4_kv_model/data_e55.json $P/gemma4_kv_scaletie/scales_e54.json" > "$LOC/pod_sha_prestart.txt"
python3 - "$MAN" "$LOC/pod_sha_prestart.txt" <<'EOF'
import json, sys
want = json.load(open(sys.argv[1]))
got = {l.split()[1].rsplit("/", 1)[1]: l.split()[0] for l in open(sys.argv[2]) if l.strip()}
bad = [n for n in want if got.get(n) != want[n]]
print(f"code hashes on the Pod: {len(want) - len(bad)} of {len(want)} match the design commit", "" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
EOF
[ $? -eq 0 ] || die "the code on the Pod is not the design commit"
g=$(ssh -n -F $C $A 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader')
echo "GPU: $g"
case "$g" in *"$GPU"*) ;; *) die "the GPU is not the planned GPU ($GPU); abort rule 1" ;; esac
START_CMD="mkdir /workspace/e55_started 2>/dev/null && env $SENV CODE=$P HF_HOME=/workspace/cache/hf VLLM_CACHE_ROOT=/workspace/cache/vllm AUTO_STOP=1 setsid nohup sh $P/gemma4_kv_model/session_e55.sh > /workspace/nohup_e55.txt 2>&1 < /dev/null &"
for k in 1 2 3; do
    ssh -f -n -F $C $A "$START_CMD" || echo "start try $k: ssh exit $?"
    perl -e 'select(undef,undef,undef,15)'
    n=$(ssh -n -F $C $A 'ps aux | grep -c "[s]ession_e55"' 2>/dev/null)
    [ "${n:-0}" -ge 1 ] && break
done
echo "session processes: ${n:-?}; $(date -u +%H:%M:%S)"
[ "${n:-0}" -ge 1 ] || die "no session process after the start"
WATCH_CMD=$(cat <<'EOF'
cat /workspace/results/e55/session.log 2>/dev/null
echo "@@@ $(ps aux | grep -c '[s]ession_e55')"
python3 -c "import json; r = json.load(open('/workspace/results/e55/run_main.json')); print('progress', r.get('status'), ' '.join(k + ':' + str(v.get('seconds')) for k, v in r.get('arms', {}).items()), 'cut', [c.get('step') for c in r.get('cut', [])])" 2>/dev/null
true
EOF
)
echo "--- watch (max $MAX s)"
t0=$(date +%s); seen=0; fails=0; ended=0; last=""
while [ $(( $(date +%s) - t0 )) -lt "$MAX" ]; do
    out=$(ssh -n -F "$C" "$A" "$WATCH_CMD" 2>/dev/null); rc=$?
    if [ $rc -ne 0 ]; then
        fails=$((fails + 1)); echo "$(date -u +%H:%M:%S) ssh failed ($fails)"; [ $fails -ge 10 ] && break
    else
        fails=0
        logp=$(printf '%s\n' "$out" | awk '/^@@@ /{exit} {print}')
        procs=$(printf '%s\n' "$out" | awk '/^@@@ /{print $2}')
        prog=$(printf '%s\n' "$out" | grep '^progress' | tail -1)
        n=$(printf '%s\n' "$logp" | grep -c .)
        if [ "$n" -gt "$seen" ]; then printf '%s\n' "$logp" | grep . | sed -n "$((seen + 1)),${n}p"; seen=$n; fi
        if [ -n "$prog" ] && [ "$prog" != "$last" ]; then echo "$(date -u +%H:%M:%S) $prog"; last=$prog; fi
        [ "$seen" -eq 0 ] && [ $(( $(date +%s) - t0 )) -gt 180 ] && die "no session.log 3 min after the start"
        printf '%s\n' "$logp" | grep -qE '^[0-9]{2}:[0-9]{2}:[0-9]{2} done after' && { ended=1; break; }   # not "setup done after"
        [ "${procs:-1}" -eq 0 ] && { echo "$(date -u +%H:%M:%S) SESSION PROCESS GONE without 'done after'"; break; }
    fi
    perl -e 'select(undef,undef,undef,20)'
done
echo "--- watch ended: session ended=$ended after $(( $(date +%s) - t0 )) s; $(date -u +%H:%M:%S)"
c0=$(date +%s); copied=0
for k in 1 2 3; do
    rsync -rlt --timeout=60 -e "ssh -F $C" "$A:$REM/" "$LOC/e55/" && { copied=1; break; }
    echo "copy try $k failed"; perl -e 'select(undef,undef,undef,10)'
done
rsync -rlt --timeout=60 -e "ssh -F $C" "$A:/workspace/boot_watchdog.log" "$A:/workspace/nohup_e55.txt" "$LOC/" 2>/dev/null
[ $copied = 1 ] && echo "RES copied after $(( $(date +%s) - c0 )) s"
ssh -n -F "$C" "$A" "find $REM -type f -exec sha256sum {} + | sed 's|$REM/||' | sort -k 2" > "$LOC/remote_sha.txt" 2>/dev/null
find "$LOC/e55" -type f -exec shasum -a 256 {} + | sed "s|$LOC/e55/||" | sort -k 2 > "$LOC/local_sha.txt"
if [ -s "$LOC/remote_sha.txt" ] && diff "$LOC/remote_sha.txt" "$LOC/local_sha.txt" > /dev/null; then
    echo "RES checksums match ($(wc -l < "$LOC/local_sha.txt" | tr -d ' ') files, $(du -sk "$LOC/e55" | cut -f1) KB)"
else
    echo "RES CHECKSUM MISMATCH or copy incomplete (a log line can be added during the copy)"
fi
[ -f "$LOC/e55/decision.txt" ] && { echo "--- decision.txt (last 3 lines)"; tail -3 "$LOC/e55/decision.txt" | cut -c1-1500; }
if [ "$WAITSTOP" = 1 ] && [ $ended = 1 ]; then
    w0=$(date +%s); st=""
    while [ $(( $(date +%s) - w0 )) -lt 480 ]; do
        st=$(pod_query desiredStatus 2>/dev/null)
        case "$st" in *EXITED*) break ;; esac
        perl -e 'select(undef,undef,undef,15)'
    done
    echo "AUTO_STOP check: pod $st $(( $(date +%s) - w0 )) s after the copy; $(date -u +%H:%M:%S)"
fi
echo "run script end $(date -u +%H:%M:%S): DELETE THE POD (pods.py check, delete-pod, pods.py end)"
