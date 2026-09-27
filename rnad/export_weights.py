"""方策・価値ネットの重みを tsuitate-bot の Rust 推論（`src/rl/policy_net.rs`）用に書き出す。

    .venv/bin/python rnad/export_weights.py out_rnad/bc/checkpoint.pt out_rnad/bc/policy.bin

形式（リトルエンディアン）:
    b"TSRLPV01" | u32 ヘッダ長 | ヘッダ JSON（planes/channels/blocks と tensors の名前・形）| f32 の並び
BatchNorm は直前の畳み込みへ畳み込んで書く（Rust 側は畳み込み＋バイアスだけを持つ）。

`--test-vector <path>` を付けると、検証用の入力と PyTorch の出力（eval モード）を JSON で書く。
Rust のテストがこれと突き合わせる。`--tiny` は学習なしの小さいネット（BatchNorm の統計も
乱数にして畳み込みの経路を検査する）を作って書き出す（tsuitate-bot のテスト用フィクスチャ）。
"""

import argparse
import json
import pathlib
import struct

import numpy as np
import torch

from model import PolicyValueNet

MAGIC = b"TSRLPV01"


def fold(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale[:, None, None, None]
    b = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        b = b + conv.bias * scale
    return w, b


def tensors(net: PolicyValueNet):
    out = []
    w, b = fold(net.stem, net.stem_bn)
    out += [("stem.w", w), ("stem.b", b)]
    for i, blk in enumerate(net.blocks):
        for j, (c, bn) in enumerate([(blk.c1, blk.b1), (blk.c2, blk.b2)]):
            w, b = fold(c, bn)
            out += [(f"block{i}.c{j + 1}.w", w), (f"block{i}.c{j + 1}.b", b)]
    w, b = fold(net.p1, net.p1_bn)
    out += [("p1.w", w.flatten(1)), ("p1.b", b)]
    out += [("p2.w", net.p2.weight.flatten(1)), ("p2.b", net.p2.bias)]
    w, b = fold(net.v1, net.v1_bn)
    out += [("v1.w", w.flatten(1)), ("v1.b", b)]
    out += [("v_fc1.w", net.v_fc1.weight), ("v_fc1.b", net.v_fc1.bias)]
    out += [("v_fc2.w", net.v_fc2.weight), ("v_fc2.b", net.v_fc2.bias)]
    return [(n, t.detach().float().cpu().contiguous()) for n, t in out]


def write(net: PolicyValueNet, path: pathlib.Path) -> None:
    net.eval()
    ts = tensors(net)
    header = {**net.config, "tensors": [{"name": n, "shape": list(t.shape)} for n, t in ts]}
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hb)))
        f.write(hb)
        for _, t in ts:
            f.write(t.numpy().astype("<f4").tobytes())


def write_test_vector(net: PolicyValueNet, path: pathlib.Path, n: int = 3, seed: int = 0) -> None:
    net.eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, net.config["planes"], 9, 9), generator=g)
    with torch.no_grad():
        logits, v = net(x)
    path.write_text(json.dumps({
        "input": x.flatten(1).tolist(),
        "logits": logits.tolist(),
        "value": v.tolist(),
    }))


def tiny_net(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    net = PolicyValueNet(channels=8, blocks=1)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 1.5)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", help="train_bc.py の checkpoint.pt（--tiny なら不要）")
    ap.add_argument("out")
    ap.add_argument("--test-vector")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    if args.tiny:
        net = tiny_net()
    else:
        ck = torch.load(args.checkpoint, map_location="cpu")
        net = PolicyValueNet(ck["config"]["channels"], ck["config"]["blocks"], ck["config"]["planes"])
        net.load_state_dict(ck["state_dict"])
    write(net, pathlib.Path(args.out))
    if args.test_vector:
        write_test_vector(net, pathlib.Path(args.test_vector))
    print(f"書き出し: {args.out}（{net.config}）")


if __name__ == "__main__":
    main()
