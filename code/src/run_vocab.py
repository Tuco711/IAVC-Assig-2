"""Train extra BoVW vocabularies and encode all images in a single pass.

Same pipeline as ``src.train`` + ``src.extract`` (identical descriptors, k-means settings, 2x2 tiling), but
descriptors are computed once per image and re-used for every K (no huge feature cache), and it logs costs:
codebook fitting time per K, descriptor time and assignment/tiling time per image.

    python -m src.run_vocab train  -k 32 64 256 1024
    python -m src.run_vocab encode -k 32 64 256 1024 --shard 0 --shards 2
    python -m src.run_vocab merge  -k 32 64 256 1024
    python -m src.run_vocab timing          # extraction cost of provided BoVW / VLAD / FV encoders
Results: data/encoders/bovw_K*_C0.npz, data/encodings/bovw_K*_C0/, results/part3_vocab_costs.csv.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from .encoders import ImageEncoder
from .features import extractFeatures
from .train import sampleLocalFeatures, train_encoder
from .utils import loadDataTables, loadSamplesTable

ROOT = Path(__file__).resolve().parents[2] / "data"
TMP = ROOT / "tmp_vocab"
RES = Path(__file__).resolve().parents[1] / "results"
LOG = RES / "part3_vocab_costs.json"


def log(key, value):
    d = json.loads(LOG.read_text()) if LOG.exists() else {}
    d[key] = value
    LOG.write_text(json.dumps(d, indent=1))


def train(ks, seed=42, per_class=150, per_image=400):
    np.random.seed(seed)
    df = loadSamplesTable(ROOT / "splits", ROOT / "images", subset="train", per_class=per_class, random_state=seed)
    t = time.perf_counter()
    descs = sampleLocalFeatures(df, num_samples=per_image, seed=seed, cache_dir="")
    log("sample_descriptors_s", time.perf_counter() - t)
    log("n_sampled_descriptors", int(descs.shape[0]))
    for k in ks:
        out = ROOT / "encoders" / f"bovw_K{k}_C0.npz"
        if out.exists():
            continue
        t = time.perf_counter()
        enc = train_encoder(descs, encoder_type="bovw", num_words=k, pca_dim=0, seed=seed)
        log(f"codebook_fit_s_K{k}", time.perf_counter() - t)
        enc.update(num_samples=descs.shape[0], num_samples_per_image=per_image, num_images=len(df), seed=seed, cache_dir="")
        np.savez(out, **enc)
        print("trained", k, flush=True)


def image_list():
    df = loadDataTables(str(ROOT / "splits"), str(ROOT / "images"))
    return df, sorted(df.filename.unique())


def encode(ks, shard, shards):
    TMP.mkdir(exist_ok=True)
    _, names = image_list()
    encs = {k: ImageEncoder(ROOT / "encoders" / f"bovw_K{k}_C0.npz") for k in ks}
    t_desc, t_enc, done = 0.0, {k: 0.0 for k in ks}, 0
    todo = names[shard::shards]
    for block in range(0, len(todo), 200):
        out = TMP / f"s{shard}_{block}.npz"
        if out.exists():
            continue
        batch, H = todo[block:block + 200], {k: [] for k in ks}
        for nm in batch:
            t = time.perf_counter()
            kp, de, sz = extractFeatures(ROOT / "images" / f"{nm}.jpg", cache_dir="")
            t_desc += time.perf_counter() - t
            for k in ks:
                t = time.perf_counter()
                H[k].append(encs[k].encode_tiling(kp, de, sz).astype(np.int32))
                t_enc[k] += time.perf_counter() - t
            done += 1
        np.savez(out, names=np.array(batch), t_desc=t_desc, done=done, **{f"K{k}": np.stack(v) for k, v in H.items()},
                 **{f"t_K{k}": v for k, v in t_enc.items()})
        print(f"shard {shard}: {done}/{len(todo)} desc {1e3 * t_desc / done:.0f} ms " +
              " ".join(f"K{k} {1e3 * t_enc[k] / done:.0f}ms" for k in ks), flush=True)


def merge(ks):
    df, names = image_list()
    files = sorted(TMP.glob("s*.npz"))
    H, last = {k: {} for k in ks}, {}
    for f in files:
        d = np.load(f)
        for k in ks:
            H[k].update(zip(d["names"].tolist(), d[f"K{k}"]))
        s = f.name.split("_")[0]
        if s not in last or int(d["done"]) > last[s]["done"]:
            last[s] = dict(done=int(d["done"]), t_desc=float(d["t_desc"]), **{f"K{k}": float(d[f"t_K{k}"]) for k in ks})
    assert len(H[ks[0]]) == len(names), (len(H[ks[0]]), len(names))
    n = sum(v["done"] for v in last.values())
    log("descriptor_ms_per_img", 1e3 * sum(v["t_desc"] for v in last.values()) / n)
    for k in ks:
        log(f"encode_ms_per_img_K{k}", 1e3 * sum(v[f"K{k}"] for v in last.values()) / n)
        out = ROOT / "encodings" / f"bovw_K{k}_C0"
        out.mkdir(exist_ok=True)
        for (c, s), r in df.groupby(["category", "subset"]):
            fn = r.filename.tolist()
            np.savez_compressed(out / f"{c}_{s}.npz", file_names=np.array(fn), histograms=np.stack([H[k][x] for x in fn]))
    print("merged", n)


def timing(n=60):
    _, names = image_list()
    names = list(np.random.default_rng(0).choice(names, n, replace=False))
    t = time.perf_counter()
    F = [extractFeatures(ROOT / "images" / f"{x}.jpg", cache_dir="") for x in names]
    log("timing_descriptor_ms_per_img", 1e3 * (time.perf_counter() - t) / n)
    for e in ("bovw_K512_C0", "vlad_K32_C80", "fv_K32_C80"):
        enc = ImageEncoder(ROOT / "encoders" / f"{e}.npz")
        t = time.perf_counter()
        for kp, de, sz in F:
            enc.encode_tiling(kp, de, sz)
        log(f"timing_encode_ms_per_img_{e}", 1e3 * (time.perf_counter() - t) / n)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "encode", "merge", "timing"])
    p.add_argument("-k", type=int, nargs="+", default=[32, 64, 256, 1024])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    {"train": lambda: train(a.k), "encode": lambda: encode(a.k, a.shard, a.shards),
     "merge": lambda: merge(a.k), "timing": timing}[a.cmd]()
