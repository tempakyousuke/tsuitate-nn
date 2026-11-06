"""書き出し（export_weights.py の CLI）の `--which` の検査。tsuitate_rl は要らない。

    .venv/bin/python rnad/tests/test_export_weights.py

- MMD の checkpoint（net / target / magnet_net）から、選んだネットがそのまま書き出される（read で戻すと出力が同じ）
- `--which` を省くと target（学習が書き出す policy_<step>.bin と同じ）。train_bc.py の checkpoint（state_dict）は従来どおり
- 無いネットを選ぶと止まる
"""

import pathlib
import subprocess
import sys
import tempfile

import torch

HERE = pathlib.Path(__file__).resolve().parent
RNAD = HERE.parent
sys.path.insert(0, str(RNAD))
from export_weights import read  # noqa: E402
from model import NUM_PLANES_DEDUCED, PolicyValueNet  # noqa: E402


def net(seed):
    torch.manual_seed(seed)
    n = PolicyValueNet(8, 1, NUM_PLANES_DEDUCED)
    for m in n.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 1.5)
    return n.eval()


def export(ck_path, out, *extra):
    return subprocess.run([sys.executable, str(RNAD / "export_weights.py"), str(ck_path), str(out), *extra],
                          capture_output=True, text=True)


def same(a, b, x):
    with torch.no_grad():
        la, va = a(x)
        lb, vb = b(x)
    return float((la - lb).abs().max()) < 1e-4 and float((va - vb).abs().max()) < 1e-5


def main():
    nets = {"net": net(1), "target": net(2), "magnet_net": net(3)}
    x = torch.rand(5, NUM_PLANES_DEDUCED, 9, 9)
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        ck = {"algo": "mmd", "config": dict(nets["net"].config), "step": 248000,
              **{k: v.state_dict() for k, v in nets.items()}}
        torch.save(ck, d / "mmd.pt")
        for which in ("net", "target", "magnet_net"):
            r = export(d / "mmd.pt", d / f"{which}.bin", "--which", which)
            assert r.returncode == 0, r.stderr
            back = read(d / f"{which}.bin")
            assert same(back, nets[which], x), which
            others = [k for k in nets if k != which]
            assert not any(same(back, nets[k], x) for k in others), f"{which} が別のネットと同じ"
        r = export(d / "mmd.pt", d / "default.bin")
        assert r.returncode == 0, r.stderr
        assert same(read(d / "default.bin"), nets["target"], x), "既定は target"
        r = export(d / "mmd.pt", d / "bad.bin", "--which", "state_dict")
        assert r.returncode != 0 and "state_dict が無い" in (r.stderr + r.stdout), r.stderr

        bc = {"config": dict(nets["net"].config), "state_dict": nets["net"].state_dict()}
        torch.save(bc, d / "bc.pt")
        r = export(d / "bc.pt", d / "bc.bin")
        assert r.returncode == 0, r.stderr
        assert same(read(d / "bc.bin"), nets["net"], x), "train_bc.py の checkpoint は state_dict"
    print("ok: export_weights --which")


if __name__ == "__main__":
    main()
