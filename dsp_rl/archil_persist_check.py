"""Persistence check: write a line on the Archil disk in one exec, read it back in another.

Every exec runs in a fresh container, so a successful read-back proves the file lives on
the disk and not in the container. Needs ARCHIL_API_KEY (a key-... API key from
https://console.archil.com/api-keys, in dsp_rl/.env or the environment); uses the disk in
ARCHIL_DISK_ID or creates one and records its id in .env. The one-time mount token is
never printed.

Run from the dsp_rl directory:
    python archil_persist_check.py
Exit codes: 0 = persisted, 1 = write/read mismatch, 2 = no API key / disk id, 3 = API error.
"""
import datetime, sys, uuid
import archil
import archil_store as store

DISK_NAME = "dsp-rl-runs"


def main():
    try:
        disk, created = store.ensure_disk(DISK_NAME)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except archil.ArchilApiError as e:
        print(f"error: the Archil API rejected the request ({e}).\n"
              "       A 401 usually means ARCHIL_API_KEY is a disk/mount token (adt_...), or a key\n"
              "       from another region: use a key-... API key from\n"
              "       https://console.archil.com/api-keys in an AWS region.", file=sys.stderr)
        return 3
    if created:
        store.save_disk_id(disk.id)
        print(f"created disk {disk.name} ({disk.id})")
    print(f"disk {disk.name} ({disk.id}), region {store.region()}")

    job = f"jobs/{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    line = f"archil-persist-check {job} {uuid.uuid4().hex}"

    # first exec: a fresh per-job directory on the disk (shared mount -> own subdir)
    w = store.run(f"mkdir -p {job} && printf '%s\\n' '{line}' > {job}/marker.txt && cat {job}/marker.txt",
                  disk=disk)
    if w.exit_code != 0:
        print(f"error: the write exec failed (exit {w.exit_code}):\n{w.stderr.strip()}", file=sys.stderr)
        return 1

    # second, separate exec: fresh container, only the disk can carry the file across
    r = store.run(f"cat {job}/marker.txt", disk=disk)
    if r.exit_code != 0:
        print(f"error: the read exec failed (exit {r.exit_code}):\n{r.stderr.strip()}", file=sys.stderr)
        return 1

    print(f"wrote:     {line}")
    print(f"read back: {r.stdout.strip()}")
    if line not in r.stdout:
        print(f"FAIL: {job}/marker.txt did not survive between execs", file=sys.stderr)
        return 1
    print(f"OK: {job}/marker.txt persisted across two execs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
