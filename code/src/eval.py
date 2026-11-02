"""Shared evaluation helpers: normalisation, validation split, SVM fit/eval with costs, retrieval metrics.

Protocol: the provided ``val`` split is the held-out TEST set (used once, for reporting).
Hyper-parameters (C, normalisation, ...) are chosen on a ``dev`` split carved out of ``train``.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

from . import utils

CATEGORIES = ["motorbike", "aeroplane", "person", "car", "horse"]
CS = (0.1, 1, 10, 100, 1000)


# ---------------------------------------------------------------- normalisation
def l1(x):
    return x / (np.abs(x).sum(1, keepdims=True) + 1e-10)


def l2(x):
    return x / (np.sqrt((x ** 2).sum(1, keepdims=True)) + 1e-10)


def hellinger(x):
    """Explicit Hellinger feature map: sqrt of L1-normalised histogram (already unit L2)."""
    return np.sqrt(l1(x))


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def intra_norm(x, k):
    """VLAD intra-normalisation: L2-normalise each of the k blocks of each row."""
    n = x.shape[0]
    return l2(x.reshape(n * k, -1)).reshape(n, -1)


NORMS = {
    "none": lambda x: x,
    "l1": l1,
    "l2": l2,
    "hellinger": hellinger,                    # sqrt(L1(h)); L2 norm is 1 by construction
    "ssqrt_l2": lambda x: l2(signed_sqrt(x)),  # power-norm used for VLAD / FV
}


# ---------------------------------------------------------------- data
def load(encoding, category, root="data"):
    """Return dict(train=(X,y,names), test=(X,y,names)); ``test`` is the provided val split."""
    return {s: utils.loadEncodingSubset(category, encoding, subset=ss, root_dir=root)
            for s, ss in (("train", "train"), ("test", "val"))}


def train_dev_split(X, y, names, dev_frac=0.2, seed=0):
    """Stratified train/dev split of the training data (model selection only)."""
    idx = np.arange(len(y))
    tr, dv = train_test_split(idx, test_size=dev_frac, stratify=y, random_state=seed)
    names = np.asarray(names)
    return (X[tr], y[tr], names[tr]), (X[dv], y[dv], names[dv])


def subsample(X, y, fraction, seed):
    """Seeded random subset (fraction<1 or an absolute count); retries to keep both classes."""
    n = len(y)
    k = n if np.isinf(fraction) else int(np.ceil(fraction * n)) if fraction <= 1 else int(fraction)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        sel = rng.choice(n, size=min(k, n), replace=False)
        if (y[sel] > 0).any() and (y[sel] < 0).any():
            break
    return X[sel], y[sel]


# ---------------------------------------------------------------- SVM
def fit(X, y, C=10, **kw):
    """Train the linear SVM and return (W, b, train_seconds)."""
    t = time.perf_counter()
    W, b = utils.train_svm(X, y, C=C, **kw)
    return W, b, time.perf_counter() - t


def score(W, b, X):
    return X @ W + b


def ap(y, s):
    return average_precision_score(y > 0, s)


def select_C(Xtr, ytr, Xdev, ydev, Cs=CS):
    """Grid over C on (train, dev). Returns best C and a DataFrame with train/dev AP."""
    rows = []
    for C in Cs:
        W, b, _ = fit(Xtr, ytr, C=C)
        rows.append(dict(C=C, train_ap=ap(ytr, score(W, b, Xtr)), dev_ap=ap(ydev, score(W, b, Xdev))))
    df = pd.DataFrame(rows)
    return float(df.loc[df.dev_ap.idxmax(), "C"]), df


# ---------------------------------------------------------------- metrics
def precision_recall_at(y, s, thr=0.0):
    pred = s > thr
    tp = int((pred & (y > 0)).sum())
    return dict(precision=tp / max(pred.sum(), 1), recall=tp / max((y > 0).sum(), 1))


def precision_at_k(y, s, k):
    return float((y[np.argsort(-s)[:k]] > 0).mean())


def recall_at_k(y, s, k):
    return float((y[np.argsort(-s)[:k]] > 0).sum() / max((y > 0).sum(), 1))


def multiclass_confusion(scores, true_idx, labels):
    """scores: (N, n_classes). Returns DataFrame confusion matrix (rows=true, cols=pred)."""
    cm = confusion_matrix(true_idx, scores.argmax(1), labels=range(len(labels)))
    return pd.DataFrame(cm, index=labels, columns=labels)


# ---------------------------------------------------------------- full experiment
def run(encoding, category, norm="l2", C=None, spatial=True, fraction=np.inf, seed=0, root="data",
        dev_frac=0.2, Cs=CS, pca=0, loader=None):
    """One one-vs-rest experiment. C=None -> chosen on the dev split (never on test).

    The final model is refit on the (sub-sampled) full train split and evaluated once on test.
    ``pca`` > 0 fits PCA on the (normalised) train features only. Returns a dict with dev AP (at the
    chosen C), test AP, C, dim, PCA/train time, inference ms/img and test-feature storage.
    """
    from sklearn.decomposition import PCA
    from .encoders import removeSpatialInformation
    d = loader(category) if loader else load(encoding, category, root)
    f = NORMS[norm]
    base = f if (spatial or loader) else (lambda X: f(removeSpatialInformation(X)))
    Xtr, ytr, ntr = d["train"]
    t_pca, prep = 0.0, base
    if pca:
        t = time.perf_counter()
        p = PCA(pca, random_state=seed).fit(base(Xtr))
        t_pca = time.perf_counter() - t
        prep = lambda X: p.transform(base(X))
    dev_ap = np.nan
    if C is None:
        (Xt, yt, _), (Xd, yd, _) = train_dev_split(Xtr, ytr, ntr, dev_frac=dev_frac, seed=seed)
        C, sel = select_C(*subsample(prep(Xt), yt, fraction, seed), prep(Xd), yd, Cs)
        dev_ap = sel.dev_ap.max()
    Xf, yf = subsample(prep(Xtr), ytr, fraction, seed)
    W, b, t_fit = fit(Xf, yf, C=C)
    Xte, yte, _ = d["test"]
    Xte = prep(Xte)
    t = time.perf_counter()
    s = score(W, b, Xte)
    t_inf = (time.perf_counter() - t) / len(yte)
    return dict(encoding=encoding, category=category, norm=norm, spatial=spatial, fraction=fraction,
                seed=seed, pca=pca, C=C, dim=Xte.shape[1], n_train=len(yf), pca_s=t_pca, train_s=t_fit,
                infer_ms_per_img=1e3 * t_inf, test_MB=Xte.astype(np.float32).nbytes / 2**20,
                dev_ap=dev_ap, test_ap=ap(yte, s), **precision_recall_at(yte, s))


# ---------------------------------------------------------------- generic (per-image feature matrix) tasks
ALL_CATS = ["background"] + CATEGORIES


def split_table(root="data"):
    """One row per unique image: ``name``, ``split`` (train/val) and a boolean column per category."""
    rows = {}
    for sp in ("train", "val"):
        for c in ALL_CATS:
            for l in open(f"{root}/splits/{c}_{sp}.txt"):
                r = rows.setdefault(l.strip(), dict(name=l.strip(), split=sp, **{k: False for k in ALL_CATS}))
                r[c] = True
    t = pd.DataFrame(rows.values()).sort_values("name").reset_index(drop=True)
    t["single"] = t[ALL_CATS].sum(1) == 1
    return t


def task_idx(table, category, split):
    """Row indices and +1/-1 labels of the one-vs-background task for ``category`` in ``split``."""
    m = (table.split == split).values
    pos, neg = np.where(m & table[category].values)[0], np.where(m & table["background"].values)[0]
    return np.r_[pos, neg], np.r_[np.ones(len(pos), int), -np.ones(len(neg), int)]


def make_loader(F, table):
    """Loader for ``run``: F is an (N, D) feature matrix aligned with ``table`` rows."""
    def loader(cat):
        out = {}
        for sp, key in (("train", "train"), ("val", "test")):
            idx, y = task_idx(table, cat, sp)
            out[key] = (F[idx], y, table.name.values[idx])
        return out
    return loader


def ovr_scores(F, table, Cs, norm="l2", Fq=None, split="val", only_single=True):
    """Train one OvR linear SVM per category on train and score a set of images.

    Cs: dict category -> C. Returns (scores (n_eval, 5), true category index, eval row indices).
    """
    f, ld = NORMS[norm], make_loader(F, table)
    m = (table.split == split).values & (table.single.values if only_single else True) & ~table.background.values
    rows = np.where(m)[0]
    Fq = F if Fq is None else Fq
    S = []
    for c in CATEGORIES:
        Xtr, ytr, _ = ld(c)["train"]
        W, b, _ = fit(f(Xtr), ytr, C=Cs[c])
        S.append(score(W, b, f(Fq[rows])))
    true = table.loc[rows, CATEGORIES].values.argmax(1)
    return np.stack(S, 1), true, rows


def macro_f1(scores, true):
    from sklearn.metrics import f1_score
    return f1_score(true, scores.argmax(1), average="macro")


if __name__ == "__main__":
    y = np.array([1, 1, -1, -1])
    assert ap(y, np.array([4., 3, 2, 1])) == 1.0
    assert abs(np.linalg.norm(hellinger(np.random.rand(3, 9)), axis=1) - 1).max() < 1e-6
    assert precision_at_k(y, np.array([4., 3, 2, 1]), 2) == 1.0
    print("eval self-check ok")
