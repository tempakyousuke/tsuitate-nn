"""模倣学習（M2a）の教師: GitHub Actions の対局記録を取ってくる。

既定は日次の Mine tsume（現行 estimator と凍結版の自己対局 160局/日、記録は 90日保持）。

    .venv/bin/python rnad/fetch_records.py --runs 30

`data/rl_records/<run_id>/` に JSONL を置く。run の全 artifact を取り終えたら `.complete`
（取った artifact 名の一覧）を書き、**それがある run だけを取得済みとして飛ばす**。
途中で切れた run（`.complete` なし）は消して取り直す。
tsuitate-bot 側の `tsuitate_rl.RecordDataset` が読む。
"""

import argparse
import json
import pathlib
import shutil
import subprocess

REPO = "tempakyousuke/tsuitate-bot"
ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "rl_records"


def gh_json(*args: str):
    return json.loads(subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default="mine-tsume.yml")
    ap.add_argument("--runs", type=int, default=30, help="新しい順にこの数の run を取る")
    ap.add_argument("--artifact-prefix", default="arena-records-")
    args = ap.parse_args()

    runs = gh_json(
        "run", "list", "--repo", REPO, "--workflow", args.workflow, "--status", "success",
        "--limit", str(args.runs), "--json", "databaseId,createdAt",
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run_id = str(run["databaseId"])
        dest = ROOT / run_id
        if (dest / ".complete").exists():
            print(f"{run_id}: 取得済み")
            continue
        if dest.exists():
            print(f"{run_id}: 途中で切れていたので取り直す")
            shutil.rmtree(dest)
        arts = gh_json("api", f"repos/{REPO}/actions/runs/{run_id}/artifacts", "--paginate")
        names = [a["name"] for a in arts["artifacts"]
                 if a["name"].startswith(args.artifact_prefix) and not a["expired"]]
        if not names:
            print(f"{run_id}: 記録の artifact なし（期限切れ？）")
            continue
        cmd = ["gh", "run", "download", run_id, "--repo", REPO, "-D", str(dest)]
        for n in names:
            cmd += ["-n", n]
        subprocess.run(cmd, check=True)
        missing = [n for n in names if not any((dest / n).rglob("*.jsonl"))]
        if missing:
            raise SystemExit(f"{run_id}: 中身の無い artifact がある: {missing}")
        (dest / ".complete").write_text("\n".join(sorted(names)) + "\n")
        print(f"{run_id} ({run['createdAt'][:10]}): {len(list(dest.rglob('*.jsonl')))} 局")


if __name__ == "__main__":
    main()
