"""Run a command and record wall time and child-process resource use on Linux."""
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

destination = Path(sys.argv[1])
command = sys.argv[2:]
started_at = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
completed = subprocess.run(command)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
destination.write_text(json.dumps({
    "command": command,
    "started_at": started_at,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "elapsed_seconds": time.perf_counter() - started,
    "exit_code": completed.returncode,
    "user_cpu_seconds": usage.ru_utime,
    "system_cpu_seconds": usage.ru_stime,
    "max_resident_memory_kib": usage.ru_maxrss,
}, indent=2) + "\n")
sys.exit(completed.returncode)
