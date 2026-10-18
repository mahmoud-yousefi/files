"""Server-side Archil disk: run artifacts that must outlive a single process.

The API key is read from ARCHIL_API_KEY (or dsp_rl/.env, see .env.example) and the disk
from ARCHIL_DISK_ID; nothing is hardcoded and nothing here is browser-visible, so keep
this module on the training/notebook side only.

    import archil_store as store
    print(store.run("ls -la").stdout)                      # one shell command, cwd = /mnt/data
    store.write_text("runs/er20_s0/log.csv", open("runs/er20_s0/log.csv").read())
    print(store.read_text("runs/er20_s0/log.csv"))

Each exec runs in a fresh container: files on the disk persist, container state does not.
Exec mounts are shared, so write into a fresh per-job subdirectory (see
archil_persist_check.py) or run `archil checkout <path>` first.
"""
import os, pathlib
import archil

DEFAULT_REGION = "aws-us-east-1"
ENV_FILE = pathlib.Path(__file__).with_name(".env")
_configured = False


def load_env(path=ENV_FILE):
    """Read a KEY=VALUE .env file; real environment variables always win."""
    path = pathlib.Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def region():
    load_env()
    return os.environ.get("ARCHIL_REGION") or DEFAULT_REGION


def disk_id():
    load_env()
    return os.environ.get("ARCHIL_DISK_ID") or ""


def configure():
    """Configure the SDK from ARCHIL_API_KEY / ARCHIL_REGION (both fall back to the env)."""
    global _configured
    if _configured:
        return
    load_env()
    if not os.environ.get("ARCHIL_API_KEY"):
        raise RuntimeError(
            "ARCHIL_API_KEY is not set: put your key-... API key in dsp_rl/.env (see "
            ".env.example) or export it; a disk/mount token (adt_...) will get HTTP 401")
    archil.configure(api_key=os.environ["ARCHIL_API_KEY"], region=region())
    _configured = True


def get_disk(id=None):
    """The disk named by ARCHIL_DISK_ID (or `id`)."""
    configure()
    id = id or disk_id()
    if not id:
        raise RuntimeError(
            "ARCHIL_DISK_ID is not set: run archil_persist_check.py to create a disk, or put "
            "an existing dsk-... id in dsp_rl/.env")
    return archil.get_disk(id)


def ensure_disk(name, id=None):
    """The disk named by ARCHIL_DISK_ID (or `id`), else a new disk called `name`.

    Returns (disk, created). The one-time mount token is dropped on purpose: we never
    need it here, and it must not end up in a log.
    """
    configure()
    id = id or disk_id()
    if id:
        return archil.get_disk(id), False
    disk = archil.create_disk(name=name).disk
    return disk.wait_until_ready(), True


def save_disk_id(id, path=ENV_FILE):
    """Record the disk id in .env (and the current process) for later runs."""
    path = pathlib.Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [l for l in lines if not l.strip().startswith("ARCHIL_DISK_ID=")]
    lines.append(f"ARCHIL_DISK_ID={id}")
    path.write_text("\n".join(lines) + "\n")
    os.environ["ARCHIL_DISK_ID"] = id


def run(command, disk=None):
    """Run one shell command on the disk (stdout/stderr/exit_code/timing in the result)."""
    return (disk or get_disk()).exec(command)


def write_text(key, text, disk=None):
    """Write a file on the disk at `key` (object-style; parent dirs are created for you)."""
    (disk or get_disk()).put_object(key, text.encode())
    return key


def read_text(key, disk=None):
    """Read a file written by write_text / put_object back as a string."""
    return (disk or get_disk()).get_object(key).decode()
