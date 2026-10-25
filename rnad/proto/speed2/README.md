# rnad/proto/speed2 — 「さらに速くする」の試作（学習には使わない）

tsuitate-bot の `docs/rl-learning-speed.md`「さらに速くする」で検討した、計算の削減（S1〜S3）・段階の見張り（T2）・
書き出しの平均（T3）・GPU リーグの不具合（9.8）の試作と検算。**train_rnad.py からは読まれない**（本線は変わらない）。
調べた環境（CPU 4コア・GPU なし）では CPU での一致と時間しか測れなかったので、**ゲーム用 PC（RTX 3070）で確かめるために残す**。

> **2026-10-25 追記**: S1〜S3 は本体に入った（既定オフのフラグ `--fast-collect py|rust` / `--device-batch` / `--fused`。
> rl-env には `VecEnv.observe_alive_f16_csr` が入ったので F3 の差分はもう当てなくてよい）。本体の版は試作と違い、
> fused も勾配をチャンクごとに取っておいて従来と同じ順に足すので**ビット単位で同じ**（試作は相対誤差 1e-8）。
> 3070 での計時と一致の確認は `rnad/bench_update.py --collect-arms ... --update-arms ...`（rnad/README.md）。
> ここの F1〜F5 は経緯の記録として残す

- どれも tsuitate-nn のリポジトリの根から `.venv/bin/python rnad/proto/speed2/<ファイル>` で動かす。`proto_common.py` が
  `rnad/` を読めるようにし、重みの既定は `../tsuitate-bot/models/rl_deduce_v7.bin`（学習状態を作る `make_learner` は
  rl_deduce_v5〜v7 を使う。本線と同じ η 0.2・β 16・lr 1e-4・Δm 10,000）
- ファイル名は `rnad/` のモジュール（model・rnad・speed など）と重ならないようにしてある（ここのファイルが先に読まれると
  本物が隠れるため）
- 学習ログだけで見られる事前確認（律速・停止・段階の平らさ g）と、書き出しの KL の曲線は、試作ではなく
  `rnad/stage_log.py`・`rnad/stage_profile.py`・`rnad/kl_versions.py`（rnad/README.md の表）

| 記号 | ファイル | 何を示すか |
| --- | --- | --- |
| F1 | `fast_collect.py` | 収集の CPU 経路を軽くした `collect_fast`（S1(a)。生きている局を1回だけ抜く・fp16 化と CSR 化を1回だけ・入玉の記帳をベクトル化。`rust_alive=True` は F3 の Rust の観測を使う S1(b)）。単独では動かさない（F2 が呼ぶ） |
| F2 | `verify_collect.py`（＋`prof_collect.py`） | 元の `rnad.collect` と F1 の batch が**ビット単位で同じ**かと、腕ごとの壁時計・部品の秒。`prof_collect.py` は部品ごとに計時する `rnad.collect` の写し |
| F2b | `box_bench.py` | 3070 の H2D（pageable / pinned、観測 float32 / fp16・密なマスク・CSR）、update 側の観測の送り方2通り、生きている局数ごとの1手番の GPU 側の秒 |
| F3 | `observe_alive.diff` | tsuitate-bot の rl-env に `observe_alive` / `observe_alive_f16_csr`（生きている局だけを観測。後者は fp16＋CSR で返す）と計測用の `bench_parts` を足す差分 |
| F4 | `s2lite_equiv.py` | S2-lite（batch を更新の頭で1回だけデバイスへ送り、更新中の同期を除いた update）が本物の update と**ビット単位で同じ**か。`--profile` で転送と同期の回数を数える |
| F5 | `fused_update.py` | pass 1 の online 順伝播を pass 2 に融合した update（S3）の勾配が本物と一致するかと、1回の秒 |
| F6 | `stage_monitor_and_ramp.diff` | tsuitate-nn の rnad.py / train_rnad.py に `--schedule-ramp`（α の補間の長さ。既定 0 = 今と同じ）と段階の収束の見張り3つ（`kl_online_reg` / `kl_target_reg` / `kl_online_target`）を足す差分。テスト `rnad/tests/test_stage_ramp.py` つき |
| F8 | `recon_tail_avg.py` | 既存の書き出し（e 更新ごとの `policy_<step>.bin`）から、段階の後半の一様な平均（`--tail-avg`）を遡って作る。`--self-test` は合成データでの誤差 |
| F9 | `gpu_league_gap.py` | （本題の外）`--league-device gpu` の batch で、学習側の手番の vt に終局の報酬が届かない不具合の再現 |
| F10 | `wall_model.py` | pipeline の1更新 = max(収集, 更新) の粗い模型。試作を積んだときの見込みの秒 |

## ゲーム用 PC での回し方と、結果で決めること

本線と同じ PC で測るときは本線を止めてから（取り合いで秒が変わる）。順番は docs の 9.6（S1 → S2 → S3、F6 は今の本線へ入れてよい）。

### F2 / F1: 収集（S1(a)）

```sh
.venv/bin/python rnad/proto/speed2/verify_collect.py --device cuda --batch 768 \
    --modes orig,fast_np,fast_dev,orig,fast_np --out verify768.json
```

- 同じ env・同じ seed で全部の腕が同じ局を指すので、壁時計は完全なペアになる（腕を交互に2回ずつ並べると順番の影響も見える）
- **決めること**: 終了コード 0（全部の腕が基準とビット一致）なら、F1 の形を `rnad.collect` の「自己対局・λ=0・truth なし」の分岐として
  入れてよい（旧経路はリーグ・ボーナス・テストのために残す）。`fast_np` と `fast_dev` の速いほうを既定にする（docs の推奨は `fast_np`）。
  一致しなければ入れない（CUDA で違いが出るなら、どのキーが違うかが出る）
- 調べた環境の結果（CPU・torch 3スレッド・768局・rl_deduce_v7）: `orig` 81.6秒、`fast_np` 72.9〜77.5秒、どの腕もビット一致。
  CPU では順伝播が半分を占めるので、3070 での比は大きく出るはず（docs の見積もり: 収集 6.67秒 → Python だけで 4.24秒、推測）

### F2b: 3070 の転送の帯域

```sh
.venv/bin/python rnad/proto/speed2/box_bench.py
```

- **決めること**: WSL2 の pageable H2D の帯域（docs の推定の幅 3〜15GB/s）が分かれば、S1・S2 の利得の見積もりを直せる。
  2. で `chunk.to(dev).float()` が `chunk.to(dev, torch.float32)` より明らかに速ければ、update 側でも fp16 のまま送る（S2）

### F3: Rust で生きている局だけを観測（S1(b)）

```sh
cd ../tsuitate-bot && git apply ../tsuitate-nn/rnad/proto/speed2/observe_alive.diff && cd ../tsuitate-nn
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m ../tsuitate-bot/rl-env/Cargo.toml
.venv/bin/python rnad/proto/speed2/verify_collect.py --device cuda --batch 768 --modes orig,fast_np,fast_rs,fast_rs_np
# 戻すとき: (cd ../tsuitate-bot && git checkout rl-env/src/lib.rs) のあと、上の maturin develop をもう一度
```

- **決めること**: `fast_rs` がビット一致で、`fast_np` より速ければ S1(b) を入れる（既存の `observe` は残し、rnad.py の `MultiEnv` にも
  同じメソッドを持たせる。`bench_parts` は外す）。調べた環境で、差分を当てた rl-env をビルドして（cargo build --release 2分20秒）、
  32局で `fast_rs` / `fast_rs_np` が基準とビット一致した
- 当てた tsuitate_rl は venv のものを置き換えるので、本線を回す前に戻すか、そのまま採るかを決めておく

### F5 / F4: 更新（S3・S2）

```sh
# 768局の batch を一度作って保存し、α=1（7000）と補間中（2500）で従来と融合を比べる
.venv/bin/python rnad/proto/speed2/fused_update.py --games 768 --save-batch batch768.pt --device cuda --timing
# 同じ batch で S2-lite のビット一致と、転送・同期の回数
.venv/bin/python rnad/proto/speed2/s2lite_equiv.py --batch batch768.pt --games 768 --chunk 1024 --device cuda --profile
```

- **F5 で決めること**: 勾配の相対誤差が 1e-6 未満（CPU の実測 4〜9e-8）で、`sec_fused` が `sec_update` より α=1 と補間中の
  両方で小さければ、`--fused` として入れる（自己対局・c_ply 既定・報酬の縮めなし・補助ヘッドなしを assert する）。
  その後 bench_update.py に `packed / fused` の腕を足して本番の形で測る。調べた環境（CPU・128局・交互に2回）では
  α=1 で 24.9〜25.1秒 → 22.6〜23.0秒、補間中は 28.0〜30.2秒 → 27.8〜28.4秒（GPU で効く割合は未計測）
- **F4 で決めること**: CUDA でもビット一致（終了コード 0）で、同期の回数が約1,500回から数十回へ減れば S2-lite を update に入れる
  （予想。S1 を入れる前は収集が律速なので、効果は待ち時間に吸われる）

### F6: 段階の見張り（T2。今の本線へ入れてよい）

```sh
git apply --check rnad/proto/speed2/stage_monitor_and_ramp.diff && git apply rnad/proto/speed2/stage_monitor_and_ramp.diff
.venv/bin/python rnad/tests/test_stage_ramp.py && .venv/bin/python rnad/tests/test_rnad.py && .venv/bin/python rnad/tests/test_packed.py
```

- `--schedule-ramp` を付けなければ α・差し替え・更新の結果は今とビット単位で同じ（調べた環境で rl_deduce_v5〜v7 の学習状態・
  8局の batch・packed / 従来・α 0.5 / 1 / 差し替えの step で、ネット4つと既存のログの値が元の rnad.py と一致した）。
  学習ログに `kl_online_reg` / `kl_target_reg` / `kl_online_target` が足される
- **決めること**: `kl_online_reg` が段階の α=1 の区間のどこで平らになるか（= 段階の中の収束時間 T_r）。早く平らになるなら
  Δm を縮める根拠（stage_log.py の g と同じ問いの、方策の側の証拠）。`kl_target_reg` はターゲットの遅れで Δm=2,000 では
  平らにならないので判定に使わない。Δm を縮めるとき（T1）に、`--schedule-ramp 250` を一緒に入れるかの材料にもなる

### F8: 書き出しの平均を遡って作る（T3）

```sh
.venv/bin/python rnad/proto/speed2/recon_tail_avg.py --self-test
.venv/bin/python rnad/proto/speed2/recon_tail_avg.py --dir out_rnad/<本線の run> --stage-start 78000 --dm 10000 \
    --every 250 --out tail_recon_88000.bin
```

- 段階の後半（83,000〜88,000）に `--every` ごとの書き出しが全部残っていることが前提（`--export-every` の倍数なら 100 でも 500 でもよい）
- 調べた環境の `--self-test`: ベクトルの列で |再構成 − 一様平均| / |最後の書き出し − 一様平均| = 0.0035、小さいネットの書き出しの
  ファイル（BatchNorm を畳んだ形）を通して 0.011
- **決めること**: 作った `tail_recon_88000.bin` と `policy_88000.bin` を固定パネル（rnad/speedlab.py）で測り、平均のほうが強ければ
  `--tail-avg` を付けて、書き出し・凍結を平均で作る（学習は変わらない。docs の見込みは一度きり +0.04〜0.2pt）

### F9: GPU リーグの不具合（CPU でよい）

```sh
.venv/bin/python rnad/proto/speed2/gpu_league_gap.py
```

- 今のコードでは GPU リーグの局で vt=G の手番が各局 1〜3 だけ（自己対局の局は全手番）。`v_trace_scalars` を直したら、
  GPU リーグの局も全手番で vt=G になることをこれで確かめる（直してから、`--league-device gpu` のリーグ学習を再開する）

### F10: 壁時計の模型

```sh
.venv/bin/python rnad/proto/speed2/wall_model.py
```

- `STEPS` の各行を 3070 で測った値（`rnad/stage_log.py` の α<1 / α=1 の busy・更新の中央値、bench_update.py の単独の秒）に
  書き換えると、どの試作の後に律速が収集から更新へ入れ替わるかを引き直せる
