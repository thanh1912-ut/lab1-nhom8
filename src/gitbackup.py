"""Git auto-backup: commit & push important artifacts to GitHub from Kaggle.

Reads GITHUB_TOKEN from env (Kaggle Secrets attach as env var).
Pushes: results.json, run logs, metric PNGs (cm_best/cm_test), and best.pt of
the best run per (model, optimizer). Large files (ckpt.pt, last.pt) are
gitignored so the repo stays small.
"""
import os
import subprocess

try:
    from . import logger as L
except ImportError:
    import logger as L

MARKER = ".gitbackup_init_done"


def _run(cmd, cwd, check=True, capture=True):
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{r.stdout}\n{r.stderr}")
    return r


def _repo_root(cfg):
    # outputs dir sits inside the repo root (parent of outputs dir name)
    return os.path.abspath(os.path.join(cfg["output_dir"], os.pardir))


def _setup_remote(cfg, root):
    gb = cfg.get("git_backup", {})
    token = os.environ.get("GITHUB_TOKEN", "")
    repo_url = gb.get("repo_url", "")
    branch = gb.get("branch", "main")
    if not token:
        print(L.colorize(
            "git backup: GITHUB_TOKEN not set - push needs a token even for "
            "public repos (clone/pull does not). Skipping backup.", L.YELLOW))
        return None
    auth_url = (repo_url.replace("https://", f"https://{token}@")
                if repo_url else None)
    # ensure git identity
    _run(["git", "config", "user.email", "lab1-bot@users.noreply.github.com"],
         root, check=False)
    _run(["git", "config", "user.name", "lab1-bot"], root, check=False)
    # ensure origin
    origin = _run(["git", "remote", "get-url", "origin"], root,
                  check=False, capture=True)
    if origin.returncode != 0:
        url = auth_url or repo_url or ""
        if url:
            _run(["git", "remote", "add", "origin", url], root, check=False)
    elif auth_url:
        _run(["git", "remote", "set-url", "origin", auth_url], root,
             check=False)
    return branch


def _force_add_artifacts(cfg, root):
    """Add results.json + per-run artifacts (logs, cm PNGs, winner best.pt)."""
    out = cfg["output_dir"]
    paths = [os.path.join(out, "results.json")]
    runs_dir = os.path.join(out, "runs")
    if os.path.isdir(runs_dir):
        for rid in sorted(os.listdir(runs_dir)):
            rd = os.path.join(runs_dir, rid)
            for f in ("train.log", "cm_best.png", "cm_test.png"):
                p = os.path.join(rd, f)
                if os.path.exists(p):
                    paths.append(p)
    # winner best.pt per (model, optimizer): lowest val loss among seeds
    import json
    rp = os.path.join(out, "results.json")
    if os.path.isfile(rp):
        with open(rp, encoding="utf-8") as f:
            results = json.load(f)
        winners = {}
        for rid, h in results.items():
            if not h.get("done") or h.get("best_val_loss") is None:
                continue
            k = (h["model"], h["optimizer"])
            if k not in winners or h["best_val_loss"] < winners[k][1]:
                winners[k] = (rid, h["best_val_loss"])
        for (rid, _) in winners.values():
            p = os.path.join(runs_dir, rid, "best.pt")
            if os.path.exists(p):
                paths.append(p)
    add = [p for p in paths if os.path.exists(p)]
    if add:
        _run(["git", "add", "-f", "--"] + add, root)


def backup(cfg, results_path=None, msg=None):
    """Commit & push artifacts. Safe to call frequently; no-op if nothing changed."""
    gb = cfg.get("git_backup", {})
    if not gb.get("enabled", False):
        return False
    root = _repo_root(cfg)
    if not os.path.isdir(os.path.join(root, ".git")):
        print(L.colorize("git backup skipped: not a git repo", L.YELLOW))
        return False
    if not os.environ.get("GITHUB_TOKEN"):
        print(L.colorize(
            "git backup skipped: GITHUB_TOKEN not set. "
            "Push (write) needs a token even for public repos - "
            "add it in Kaggle Secrets to enable auto-backup.", L.YELLOW))
        return False
    try:
        branch = _setup_remote(cfg, root)
        _force_add_artifacts(cfg, root)
        r = _run(["git", "commit", "-m",
                  msg or f"backup: training artifacts {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}"],
                 root, check=False)
        if r.returncode != 0 and "nothing to commit" not in (r.stdout or "") + (r.stderr or ""):
            # nothing changed -> fine
            if "no changes" in (r.stdout or "") + (r.stderr or "") or \
               "nothing to commit" in (r.stdout or "") + (r.stderr or ""):
                return False
        p = _run(["git", "push", "origin", branch], root, check=False)
        if p.returncode == 0:
            print(L.colorize("✔ git backup pushed", L.GREEN))
            return True
        print(L.colorize(f"git push failed: {p.stderr}", L.RED))
        return False
    except Exception as e:
        print(L.colorize(f"git backup error: {e}", L.RED))
        return False
