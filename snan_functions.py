import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_ROOT = r"C:\envs\data"
DATASET = "FashionMNIST"
SNAN_OUTPUT = "astro"
NETS = {"impulse": ("FashionMNIST", "Trained_SNAN_FMNIST.mat", "Trained_SB_SNAN_FMNIST.mat"),
        "occlusion": ("MNIST", "Trained_SNAN_MNIST.mat", "Trained_SB_SNAN_MNIST.mat")}
FEATURES_DIR = "snan_features"
RESULTS_DIR = "results"

NUM_RUNS = 10
N_TRAIN_PER_CLASS = 750
N_TEST_PER_CLASS = 250
N_PARALLEL_RUNS = 2
N_THREADS_DATA = 16
USE_TF32 = True

SB_CLASSES = [0, 1, 2, 3, 4]
SB_N_PER_CLASS = 50
IMPULSE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
OCCLUSION_LEVELS = list(range(0, 80, 5))
EPOCH_CURVE_NOISE = 0.15

SNAN_SIZE = 79
OCCLUSION_SHRINK = 0.8
AUG_SQUARE = True
IMG_SIZE = 28
THRESHOLD = 50

TRAIN_RUN = 1
SEED = 0
I_APP = 10.0
PULSE_MS = 5.0
PAUSE_MS = 5.0
W_EE_0 = 1e-4
W_IE_0 = 1e-4
G_THR = 0.1
TRAIN_DTYPE = torch.float64
REPORT_EVERY = 250

TEST_PULSE_MS = 10.0
TEST_WINDOW_MS = 20.0
TEST_CA_THR = 0.15
TEST_W_EE_SCALE = 0.6
SPIKE_THR = 10
SYNC_MIN = 9
SNAN_BATCH = 250
TEST_DTYPE = torch.float32
N_EXAMPLES = 8

P = dict(
    dt=1e-4,
    WE=79, HE=79, WI=40, HI=40,
    MA=26, NA=26, ZONE=4,
    a=0.1, b=0.2, c=-65.0, d=2.0, v_spike=30.0,
    V_rest=-71.9795475755463, U_rest=-12.8842919091830,
    V_init=-65.0,
    E_exc=0.0, E_inh=-90.0, k_syn=0.2,
    N_EE=200, lam_EE=15.0, N_EI=5, lam_EI=2.0, N_IE=2000, lam_IE=80.0,
    w_EI=0.1,
    g_EE=0.007, w_EE_max=0.05, g_IE=0.007, w_IE_max=0.05, w_min=1e-4,
    tau_stdp=0.020,
    beta_f=200.0, f_step=0.5, f_max=1.0, f_star=0.3,
    alpha_glu=50.0, k_glu=600.0, A_glu=5.0, t_glu=0.060,
    F_act=0.75, F_astro=0.5, nu_ca=2.0, Ca_thr=0.15, tau_astro=0.020,
    watch_window=0.005, watch_every=0.001,
    c0=2.0, c1=0.185, v1=6.0, v2=0.11, v3=2.2, v4=0.3, v6=0.2, k1=0.5, k2=1.0, k3=0.1, k4=1.1,
    d1=0.13, d2=1.049, d3=0.9434, d5=0.082, IP3s=0.16, Tr=0.14, alpha=0.8, a2=0.14,
    dCa=0.05, dIP3=0.05,
    Ca0=0.072495, h0=0.886314, IP30=0.820204,
)

CNN_P = dict(epochs=30, batch=128, lr0=0.01, decay=0.001, momentum=0.9, N1=200, N2=100, augment_shift=5.0)
VIT_P = dict(epochs=50, batch=128, lr=5e-4, patch=4, dim=128, depth=8, heads=8, dim_head=64, mlp=256)
FILTER_P = dict(mean_size=5, median_size=4, bilat_degree=50.0, bilat_sigma=1.5)
FILTERS = ["mean", "median", "bilateral"]
CLASSIFIERS = ["CNN", "ViT"]

SMALL_SIZES = [750, 50, 20, 10, 3, 1]
SMALL_NOISE = 0.15
SMALL_EPOCHS = 50

OMNIGLOT_ROOT = r"C:\Users\Юля\Desktop\Тюкин ИЮ окклюзия большой трансформер\омниглот\обучение чистое\трансформер\data"
OMNI_SNAN_FILE = "Trained_SNAN_MNIST_103.mat"
OMNI_GEOM = dict(WE=103, HE=103, WI=51, HI=51, MA=34, NA=34, ZONE=4)
OMNI_INPUT = 105
OMNI_PATCH = 5
OMNI_SHOTS = [15, 10, 5]
OMNI_CLASSES = 10
OMNI_NOISE = 0.15
OMNI_EPOCHS = 50

NOISE_TYPE = "impulse"
TRAIN_NOISE = 0.0
AUG_MIN = None
SHRINK = None
LEVELS = IMPULSE_LEVELS


def use_dataset(name):
    global DATASET, NETS, OCCLUSION_SHRINK, RESULTS_DIR, FEATURES_DIR
    DATASET = name
    if name == "MNIST":
        NETS = {"impulse": ("MNIST", "Trained_SNAN_MNIST.mat", "Trained_SB_SNAN_MNIST.mat"),
                "occlusion": ("MNIST", "Trained_SNAN_MNIST.mat", "Trained_SB_SNAN_MNIST.mat")}
        OCCLUSION_SHRINK = None
        RESULTS_DIR, FEATURES_DIR = "results_MNIST", "snan_features_MNIST"
    else:
        NETS = {"impulse": ("FashionMNIST", "Trained_SNAN_FMNIST.mat", "Trained_SB_SNAN_FMNIST.mat"),
                "occlusion": ("MNIST", "Trained_SNAN_MNIST.mat", "Trained_SB_SNAN_MNIST.mat")}
        OCCLUSION_SHRINK = 0.8
        RESULTS_DIR, FEATURES_DIR = "results", "snan_features"


def load_mnist(root, dataset=None):
    from torchvision.datasets import MNIST, FashionMNIST
    D = FashionMNIST if (dataset or DATASET) == "FashionMNIST" else MNIST
    tr = D(root, train=True, download=True)
    te = D(root, train=False, download=True)
    return (tr.data.numpy(), tr.targets.numpy().astype(np.int64),
            te.data.numpy(), te.targets.numpy().astype(np.int64))


def pick_split(ytr, yte, n_tr, n_te, seed):
    rs = np.random.RandomState(seed)
    itr, ite = [], []
    for c in range(10):
        a = np.where(ytr == c)[0]
        rs.shuffle(a)
        itr.extend(a[:n_tr])
        b = np.where(yte == c)[0]
        rs.shuffle(b)
        ite.extend(b[:n_te])
    return np.array(itr), np.array(ite)


def shrink28(imgs):
    if not SHRINK:
        return imgs
    from PIL import Image
    n = int(IMG_SIZE * SHRINK)
    o = (IMG_SIZE - n) // 2
    out = np.zeros_like(imgs)
    for i, a in enumerate(imgs):
        out[i, o:o + n, o:o + n] = np.asarray(Image.fromarray(a).resize((n, n), Image.NEAREST))
    return out


def binarize81(imgs, n_threads=1, size=None):
    from PIL import Image
    size = size or SNAN_SIZE

    def one(a):
        b = np.asarray(Image.fromarray(a).resize((size, size), Image.BICUBIC))
        return (b >= THRESHOLD).astype(np.uint8)
    with ThreadPoolExecutor(n_threads) as ex:
        return np.stack(list(ex.map(one, imgs)))


def add_noise(B, level, rng):
    B = B.copy()
    n_pix = B.shape[1] * B.shape[2]
    k = int(round(n_pix * level))
    if k == 0:
        return B
    flat = B.reshape(len(B), -1)
    r = rng.random((len(B), n_pix), dtype=np.float32)
    ind = np.argpartition(r, k - 1, axis=1)[:, :k]
    rows = np.arange(len(B))[:, None]
    flat[rows, ind] = 1 - flat[rows, ind]
    return B


def bin_resize(img28, w, h):
    from PIL import Image
    return (np.asarray(Image.fromarray(img28).resize((int(w), int(h)), Image.BICUBIC)) >= THRESHOLD).astype(np.uint8)


def occlude(B, raw_pool, size, rng):
    B = B.copy()
    size = int(round(size))
    if size == 0:
        return B
    H, W = B.shape[1], B.shape[2]
    src = rng.integers(0, len(raw_pool), len(B))
    corner = rng.integers(0, 4, len(B))
    cache = {}
    for i in range(len(B)):
        o = cache.get(src[i])
        if o is None:
            o = cache[src[i]] = bin_resize(raw_pool[src[i]], size, size)
        r0 = 0 if corner[i] in (0, 1) else H - size
        c0 = 0 if corner[i] in (0, 2) else W - size
        B[i, r0:r0 + size, c0:c0 + size] |= o
    return B


def augment_scale_shift(raw, amin, rng):
    out = np.zeros((len(raw), SNAN_SIZE, SNAN_SIZE), np.uint8)
    w = rng.integers(int(amin), 80, len(raw))
    h = w.copy() if AUG_SQUARE else rng.integers(int(amin), 80, len(raw))
    for i in range(len(raw)):
        b = bin_resize(raw[i], w[i], h[i])
        r0 = rng.integers(0, SNAN_SIZE - b.shape[0] + 1)
        c0 = rng.integers(0, SNAN_SIZE - b.shape[1] + 1)
        out[i, r0:r0 + b.shape[0], c0:c0 + b.shape[1]] = b
    return out


def corrupt_test(Bte, raw_te, nu, seed):
    if NOISE_TYPE == "occlusion":
        return occlude(Bte, raw_te, nu, np.random.default_rng([seed, 4, int(round(nu))]))
    return add_noise(Bte, nu, np.random.default_rng([seed, 2, int(round(nu * 1000))]))


def train_key():
    if NOISE_TYPE == "occlusion":
        return float(AUG_MIN or 0)
    return round(float(TRAIN_NOISE), 4)


def make_train(Btr, raw_tr, key, seed):
    if NOISE_TYPE == "occlusion":
        if not key:
            return Btr.copy()
        return augment_scale_shift(raw_tr, key, np.random.default_rng([seed, 5, int(key)]))
    return add_noise(Btr, key, np.random.default_rng([seed, 1, int(round(key * 1000))]))


def down28(B):
    nn_idx = np.floor((np.arange(IMG_SIZE) + 0.5) * B.shape[1] / IMG_SIZE).astype(int)
    return B[:, nn_idx][:, :, nn_idx].astype(np.float32)


def to_tensor(X, dev):
    return torch.tensor(X[:, None], device=dev)


def connections(rng, Wpre, Hpre, Wpost, Hpost, Kx, Ky, n_out, lam, exclude_self):
    pre_all, post_all = [], []
    chunk = max(4 * n_out, 2000)
    for idx in range(Wpre * Hpre):
        x, y = idx % Wpre + 1, idx // Wpre + 1
        got = np.empty(0, np.int64)
        while got.size < n_out:
            r = rng.exponential(lam, chunk)
            phi = rng.uniform(0, 2 * np.pi, chunk)
            xp = np.ceil(Kx * x + r * np.cos(phi)).astype(np.int64)
            yp = np.ceil(Ky * y + r * np.sin(phi)).astype(np.int64)
            ok = (xp >= 1) & (xp <= Wpost) & (yp >= 1) & (yp <= Hpost)
            lin = (xp[ok] - 1) + (yp[ok] - 1) * Wpost
            if exclude_self:
                lin = lin[lin != idx]
            cand = np.concatenate([got, lin])
            _, first = np.unique(cand, return_index=True)
            got = cand[np.sort(first)]
        pre_all.append(np.full(n_out, idx, np.int64))
        post_all.append(got[:n_out])
    return np.concatenate(pre_all), np.concatenate(post_all)


def make_all_connections(seed):
    rng = np.random.default_rng(seed)
    WE, WI = P["WE"], P["WI"]
    K = WE / WI
    return dict(EE=connections(rng, WE, P["HE"], WE, P["HE"], 1.0, 1.0, P["N_EE"], P["lam_EE"], True),
                EI=connections(rng, WE, P["HE"], WI, P["HI"], 1 / K, 1 / K, P["N_EI"], P["lam_EI"], False),
                IE=connections(rng, WI, P["HI"], WE, P["HE"], K, K, P["N_IE"], P["lam_IE"], False))


def zone_members():
    Z, step = P["ZONE"], P["ZONE"] - 1
    mem = np.zeros((P["MA"] * P["NA"], Z * Z), np.int64)
    for n in range(P["NA"]):
        for m in range(P["MA"]):
            xs = np.arange(m * step, m * step + Z)
            ys = np.arange(n * step, n * step + Z)
            mem[m + n * P["MA"]] = (xs[:, None] + ys[None, :] * P["WE"]).ravel()
    return mem


def neuron_zone_table(mem, n_neurons):
    nz = [[] for _ in range(n_neurons)]
    for z, row in enumerate(mem):
        for n in row:
            nz[n].append(z)
    w = max(len(v) for v in nz)
    return [v + [len(mem)] * (w - len(v)) for v in nz]


def ullah_rhs(Ca, h, IP3, I_neuro, diff_Ca, diff_IP3):
    p = P
    M = IP3 / (IP3 + p["d1"])
    NM = Ca / (Ca + p["d5"])
    Ier = p["c1"] * p["v1"] * M ** 3 * NM ** 3 * h ** 3 * ((p["c0"] - Ca) / p["c1"] - Ca)
    Ileak = p["c1"] * p["v2"] * ((p["c0"] - Ca) / p["c1"] - Ca)
    Ipump = p["v3"] * Ca ** 2 / (Ca ** 2 + p["k3"] ** 2)
    Iin = p["v6"] * IP3 ** 2 / (p["k2"] ** 2 + IP3 ** 2)
    Iout = p["k1"] * Ca
    Q2 = p["d2"] * (IP3 + p["d1"]) / (IP3 + p["d3"])
    h_inf = Q2 / (Q2 + Ca)
    Tn = 1.0 / (p["a2"] * (Q2 + Ca))
    Iplc = p["v4"] * (Ca + (1 - p["alpha"]) * p["k4"]) / (Ca + p["k4"])
    dCa = Ier - Ipump + Ileak + Iin - Iout + p["dCa"] * diff_Ca
    dh = (h_inf - h) / Tn
    dIP3 = (p["IP3s"] - IP3) * p["Tr"] + Iplc + I_neuro + p["dIP3"] * diff_IP3
    return dCa, dh, dIP3


def neighbour_sum(X):
    D = torch.zeros_like(X)
    D[1:, :] += X[:-1, :] - X[1:, :]
    D[:-1, :] += X[1:, :] - X[:-1, :]
    D[:, 1:] += X[:, :-1] - X[:, 1:]
    D[:, :-1] += X[:, 1:] - X[:, :-1]
    return D


class SNAN:
    def __init__(self, C, dev, dtype):
        self.dev, self.dt_ = dev, dtype
        t = lambda a: torch.as_tensor(a, dtype=torch.long, device=dev)
        self.NE, self.NI = P["WE"] * P["HE"], P["WI"] * P["HI"]
        self.pre = {k: t(C[k][0]) for k in C}
        self.post = {k: t(C[k][1]) for k in C}
        self.w = {"EE": torch.full((len(C["EE"][0]),), W_EE_0, dtype=dtype, device=dev),
                  "IE": torch.full((len(C["IE"][0]),), W_IE_0, dtype=dtype, device=dev)}
        mem = zone_members()
        self.A, self.nZ = mem.shape
        self.members = t(mem)
        self.neuron_zones = t(neuron_zone_table(mem, self.NE))
        self.n_win = int(round(P["watch_window"] / P["dt"])) + 1
        self.watch_every = int(round(P["watch_every"] / P["dt"]))
        self.t_glu = int(round(P["t_glu"] / P["dt"]))
        self.t_astro = int(round(P["tau_astro"] / P["dt"])) + 1
        self.reset_state()

    def reset_state(self):
        dev, dt_ = self.dev, self.dt_
        z = lambda n: torch.zeros(n, dtype=dt_, device=dev)
        self.VE = torch.full((self.NE,), P["V_init"], dtype=dt_, device=dev)
        self.UE = P["b"] * self.VE.clone()
        self.VI = torch.full((self.NI,), P["V_init"], dtype=dt_, device=dev)
        self.UI = P["b"] * self.VI.clone()
        self.IsynE, self.IsynI = z(self.NE), z(self.NI)
        self.G = z(self.NE)
        self.trE, self.trI = z(self.NE), z(self.NI)
        self.fE = z(self.NE)
        sh = (P["MA"], P["NA"])
        self.Ca = torch.full(sh, P["Ca0"], dtype=dt_, device=dev)
        self.h = torch.full(sh, P["h0"], dtype=dt_, device=dev)
        self.IP3 = torch.full(sh, P["IP30"], dtype=dt_, device=dev)
        self.glu_timer = torch.zeros(self.A, dtype=torch.long, device=dev)
        self.astro_timer = torch.zeros(self.A, dtype=torch.long, device=dev)
        self.spk_ring = torch.zeros((self.n_win, self.A), dtype=torch.long, device=dev)
        self.ring_pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.step_no = torch.zeros((), dtype=torch.long, device=dev)
        self.modE = z(self.NE)
        self.acc_spk = torch.zeros((), dtype=torch.long, device=dev)
        self.acc_glu = torch.zeros(self.A, dtype=torch.bool, device=dev)
        self.acc_mod = torch.zeros(self.A, dtype=torch.bool, device=dev)

    def reset_accumulators(self):
        self.acc_spk.zero_()
        self.acc_glu.zero_()
        self.acc_mod.zero_()

    def step(self, Iapp, onset):
        p, dt = P, P["dt"]
        vs = p["v_spike"]
        fE, fI = self.VE >= vs, self.VI >= vs
        VE = torch.where(fE, p["c"], self.VE)
        UE = self.UE + p["d"] * fE
        VI = torch.where(fI, p["c"], self.VI)
        UI = self.UI + p["d"] * fI
        VEn = VE + dt * 1000 * (0.04 * VE * VE + 5 * VE + 140 + Iapp + self.IsynE - UE)
        UE = UE + dt * 1000 * p["a"] * (p["b"] * VEn - UE)
        VIn = VI + dt * 1000 * (0.04 * VI * VI + 5 * VI + 140 + self.IsynI - UI)
        UI = UI + dt * 1000 * p["a"] * (p["b"] * VIn - UI)
        VE, VI = VEn.clamp(max=vs), VIn.clamp(max=vs)
        spkE, spkI = VE >= vs, VI >= vs
        if onset:
            VE = torch.full_like(VE, p["V_rest"]); UE = torch.full_like(UE, p["U_rest"])
            VI = torch.full_like(VI, p["V_rest"]); UI = torch.full_like(UI, p["U_rest"])
        self.VE.copy_(VE); self.UE.copy_(UE); self.VI.copy_(VI); self.UI.copy_(UI)
        sE, sI = spkE.to(self.dt_), spkI.to(self.dt_)

        decay = float(np.exp(-dt / p["tau_stdp"]))
        trE_old = self.trE * decay
        trI_old = self.trI * decay
        trE_new = torch.where(spkE, 1.0, trE_old)
        trI_new = torch.where(spkI, 1.0, trI_old)
        f_old = self.fE * (1 - dt * p["beta_f"])
        pre, post = self.pre["EE"], self.post["EE"]
        dw = sE[post] * trE_new[pre] - sE[pre] * trE_old[post]
        self.w["EE"].add_(p["g_EE"] * dw).clamp_(p["w_min"], p["w_EE_max"])
        pre, post = self.pre["IE"], self.post["IE"]
        gate = (f_old < p["f_star"]).to(self.dt_)
        dw = (sE * gate)[post] * trI_new[pre] - sI[pre] * trE_old[post]
        self.w["IE"].add_(p["g_IE"] * dw).clamp_(p["w_min"], p["w_IE_max"])
        self.trE.copy_(trE_new)
        self.trI.copy_(trI_new)
        self.fE.copy_(torch.where(spkE, (f_old + p["f_step"]).clamp(max=p["f_max"]), f_old))

        SE = torch.sigmoid(VE / p["k_syn"])
        self.IsynE.zero_()
        sumEI = torch.zeros(self.NI, dtype=self.dt_, device=self.dev).index_add_(
            0, self.post["EI"], p["w_EI"] * SE[self.pre["EI"]])
        self.IsynI.copy_(sumEI * (p["E_exc"] - VI))

        self.G.add_(-dt * (p["alpha_glu"] * self.G - p["k_glu"] * sE))
        act = (self.G >= G_THR)[self.members].sum(1)
        self.glu_timer.copy_(torch.where(act > p["F_act"] * self.nZ, self.t_glu, self.glu_timer))
        I_neuro = (p["A_glu"] * (self.glu_timer > 0).to(self.dt_)).reshape(P["NA"], P["MA"]).T
        self.acc_glu |= self.glu_timer > 0
        self.glu_timer.copy_((self.glu_timer - 1).clamp(min=0))
        dC, dI = neighbour_sum(self.Ca), neighbour_sum(self.IP3)
        X = (self.Ca, self.h, self.IP3)
        k1 = ullah_rhs(*X, I_neuro, dC, dI)
        k2 = ullah_rhs(*[x + dt / 2 * k for x, k in zip(X, k1)], I_neuro, dC, dI)
        k3 = ullah_rhs(*[x + dt / 2 * k for x, k in zip(X, k2)], I_neuro, dC, dI)
        k4 = ullah_rhs(*[x + dt * k for x, k in zip(X, k3)], I_neuro, dC, dI)
        for x, a, b, c, d in zip(X, k1, k2, k3, k4):
            x.add_(dt / 6 * (a + 2 * b + 2 * c + d))
        self.spk_ring.index_copy_(0, self.ring_pos, spkE[self.members].sum(1)[None])
        self.ring_pos.add_(1).remainder_(self.n_win)
        self.step_no.add_(1)
        check = (self.step_no % self.watch_every) == 0
        sync = self.spk_ring.max(0).values >= p["F_astro"] * self.nZ
        on = check & (self.Ca.T.reshape(-1) > p["Ca_thr"]) & sync
        self.astro_timer.copy_(torch.where(on, self.t_astro, self.astro_timer))
        mod_z = torch.cat([self.astro_timer > 0, torch.zeros(1, dtype=torch.bool, device=self.dev)])
        self.modE.copy_(mod_z[self.neuron_zones].any(1).to(self.dt_))
        self.acc_mod |= mod_z[:-1]
        self.astro_timer.copy_((self.astro_timer - 1).clamp(min=0))
        self.acc_spk += spkE.sum()

    def present(self, Iapp, on_steps, n_steps):
        zero = torch.zeros_like(Iapp)
        for s in range(n_steps):
            self.step(Iapp if s < on_steps else zero, s == 0)


class GraphRunner:
    def __init__(self, net, on_steps, n_steps):
        self.net = net
        self.Iapp = torch.zeros(net.NE, dtype=net.dt_, device=net.dev)
        self.args = (on_steps, n_steps)
        self.graph = None
        if net.dev.type == "cuda":
            saved = self._save()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                net.present(self.Iapp, *self.args)
            torch.cuda.current_stream().wait_stream(s)
            self._restore(saved)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                net.present(self.Iapp, *self.args)
            self._restore(saved)

    def _state(self):
        n = self.net
        return [n.VE, n.UE, n.VI, n.UI, n.IsynE, n.IsynI, n.G, n.trE, n.trI, n.fE, n.Ca, n.h, n.IP3,
                n.glu_timer, n.astro_timer, n.spk_ring, n.ring_pos, n.step_no, n.modE,
                n.acc_spk, n.acc_glu, n.acc_mod, n.w["EE"], n.w["IE"]]

    def _save(self):
        return [x.clone() for x in self._state()]

    def _restore(self, saved):
        for x, s in zip(self._state(), saved):
            x.copy_(s)

    def run(self, Iapp):
        self.Iapp.copy_(Iapp)
        if self.graph is not None:
            self.graph.replay()
        else:
            self.net.present(self.Iapp, *self.args)


def select_situation(itr, labels, rng):
    sel = []
    for c in SB_CLASSES:
        idx = itr[labels[itr] == c]
        sel.extend(rng.choice(idx, SB_N_PER_CLASS, replace=False))
    return np.array(sel)


def train(out, dataset, select=None, geom=None):
    if os.path.isfile(out):
        print(f"{out} already exists")
        return
    saved = dict(P)
    P.update(geom or {})
    try:
        _train(out, dataset, select)
    finally:
        P.clear()
        P.update(saved)


def _train(out, dataset, select=None):
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_all, Ytr_all, Xte_all, Yte_all = load_mnist(MNIST_ROOT, dataset)
    itr, _ = pick_split(Ytr_all, Yte_all, N_TRAIN_PER_CLASS, N_TEST_PER_CLASS, 42 + TRAIN_RUN - 1)
    rng = np.random.default_rng(SEED + 1000 * TRAIN_RUN)
    if select is not None:
        itr = np.asarray(select(itr, Ytr_all, rng))
    order = itr[rng.permutation(len(itr))]
    imgs = binarize81(Xtr_all[order], N_THREADS_DATA, size=P["WE"])
    labels = Ytr_all[order]
    print(f"{out}: {dataset}, {len(order)} training images, classes {sorted(set(labels.tolist()))}")
    C = make_all_connections(SEED + 1000 * TRAIN_RUN)
    net = SNAN(C, dev, TRAIN_DTYPE)
    on_steps = int(round(PULSE_MS / 1000 / P["dt"]))
    per_img = on_steps + int(round(PAUSE_MS / 1000 / P["dt"]))
    Ca_end = np.zeros((len(order), P["MA"], P["NA"]), np.float32)
    diag = {k: np.zeros(len(order)) for k in ["spikes", "glu_zones", "mod_zones", "Ca_max",
                                              "Ca_mean", "wEE_mean", "wIE_mean"]}
    t_sim = time.time()
    runner = GraphRunner(net, on_steps, per_img)
    for k, img in enumerate(imgs):
        Iapp = torch.as_tensor(I_APP * img.T.reshape(-1).astype(np.float64), dtype=TRAIN_DTYPE, device=dev)
        net.reset_accumulators()
        runner.run(Iapp)
        Ca_end[k] = net.Ca.cpu().numpy()
        diag["spikes"][k] = net.acc_spk.item() / max(int(img.sum()), 1)
        diag["glu_zones"][k] = net.acc_glu.sum().item()
        diag["mod_zones"][k] = net.acc_mod.sum().item()
        diag["Ca_max"][k] = net.Ca.max().item()
        diag["Ca_mean"][k] = net.Ca.mean().item()
        diag["wEE_mean"][k] = net.w["EE"].mean().item()
        diag["wIE_mean"][k] = net.w["IE"].mean().item()
        if (k + 1) % REPORT_EVERY == 0 or k + 1 == len(order):
            el = time.time() - t_sim
            print(f"image {k + 1}/{len(order)}: Ca max {diag['Ca_max'][k]:.3f}, "
                  f"modulating zones {diag['mod_zones'][k]:.0f}  [{el:.0f} s, ~{el / (k + 1) * (len(order) - k - 1):.0f} s left]")
    D = {
        "params": {k: float(v) for k, v in P.items()},
        "EE_pre": C["EE"][0].astype(np.int32), "EE_post": C["EE"][1].astype(np.int32),
        "EI_pre": C["EI"][0].astype(np.int32), "EI_post": C["EI"][1].astype(np.int32),
        "IE_pre": C["IE"][0].astype(np.int32), "IE_post": C["IE"][1].astype(np.int32),
        "W_EE": net.w["EE"].cpu().numpy(), "W_IE": net.w["IE"].cpu().numpy(),
        "W_EI": np.full(len(C["EI"][0]), P["w_EI"]),
        "Ca": net.Ca.cpu().numpy(), "h": net.h.cpu().numpy(), "IP3": net.IP3.cpu().numpy(),
        "Ca_after_each_image": Ca_end,
        "image_index": order.astype(np.int64),
        "image_label": labels.astype(np.int64),
        **{f"diag_{k}": v for k, v in diag.items()},
    }
    sio.savemat(out, D, do_compression=True)
    print(f"Saved {out}, total time {time.time() - t0:.0f} s")
    plot_training(out)


def plot_training(fname):
    import matplotlib.pyplot as plt
    D = sio.loadmat(fname, squeeze_me=True)
    n = np.arange(1, len(D["diag_spikes"]) + 1)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    im = ax[0].imshow(D["Ca"], cmap="hot")
    ax[0].set_title("Ca$^{2+}$ after training (μM)")
    fig.colorbar(im, ax=ax[0])
    ax[1].plot(n, D["diag_Ca_max"], label="max")
    ax[1].plot(n, D["diag_Ca_mean"], label="mean")
    ax[1].axhline(P["Ca_thr"], color="k", ls="--", lw=0.8, label="[Ca]$_{thr}$")
    ax[1].set_xlabel("Image"); ax[1].set_ylabel("Ca$^{2+}$ (μM)"); ax[1].legend()
    ax[2].plot(n, D["diag_glu_zones"], label="zones with J$_{glu}$")
    ax[2].plot(n, D["diag_mod_zones"], label="modulating zones")
    ax[2].set_xlabel("Image"); ax[2].legend()
    ax[3].semilogy(n, D["diag_wEE_mean"], label="mean w$_{EE}$")
    ax[3].semilogy(n, D["diag_wIE_mean"], label="mean w$_{IE}$")
    ax[3].set_xlabel("Image"); ax[3].legend()
    for a in ax[1:]:
        a.grid(True, alpha=0.4)
    fig.tight_layout()
    f = os.path.splitext(fname)[0] + ".png"
    fig.savefig(f, dpi=150)
    plt.show()
    print(f"Saved {f}")


def train_snan(noise):
    dataset, f, _ = NETS[noise]
    train(f, dataset)


def train_sb_snan(noise):
    dataset, _, f = NETS[noise]
    train(f, dataset, select=select_situation)


def load_snan(fname, dev, dtype):
    D = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
    prm = D["params"]
    Pf = {k: float(getattr(prm, k)) for k in prm._fieldnames}
    for k in ["WE", "HE", "WI", "HI", "MA", "NA", "ZONE", "N_EE", "N_EI", "N_IE"]:
        Pf[k] = int(round(Pf[k]))
    P.update(Pf)
    NE, NI = Pf["WE"] * Pf["HE"], Pf["WI"] * Pf["HI"]

    def csr(pre, post, w, n_post, n_pre):
        idx = torch.tensor(np.stack([post, pre]).astype(np.int64))
        S = torch.sparse_coo_tensor(idx, torch.tensor(np.asarray(w, float)), (n_post, n_pre)).coalesce()
        return S.to_sparse_csr().to(device=dev, dtype=dtype)

    W = dict(EE=csr(D["EE_pre"], D["EE_post"], TEST_W_EE_SCALE * np.asarray(D["W_EE"], float), NE, NE),
             IE=csr(D["IE_pre"], D["IE_post"], D["W_IE"], NE, NI),
             EI=csr(D["EI_pre"], D["EI_post"], D["W_EI"], NI, NE))
    return Pf, W, np.asarray(D["Ca"], float)


class SNANTest:
    def __init__(self, Pf, W, Ca, dev, dtype):
        self.P, self.W, self.dev, self.dtype = Pf, W, dev, dtype
        self.NE, self.NI = Pf["WE"] * Pf["HE"], Pf["WI"] * Pf["HI"]
        mem = zone_members()
        self.A, self.nZ = mem.shape
        self.members = torch.tensor(mem, device=dev)
        self.neuron_zones = torch.tensor(neuron_zone_table(mem, self.NE), device=dev)
        self.ca_ok = torch.tensor((Ca >= TEST_CA_THR).T.reshape(-1), device=dev)
        dt = Pf["dt"]
        self.n_win = int(round(Pf["watch_window"] / dt)) + 1
        self.watch_every = int(round(Pf["watch_every"] / dt))
        self.t_astro = int(round(Pf["tau_astro"] / dt)) + 1
        self.on_steps = int(round(TEST_PULSE_MS / 1000 / dt))
        self.n_steps = int(round(TEST_WINDOW_MS / 1000 / dt))

    @torch.no_grad()
    def run(self, imgs81):
        p, dt, dev, dtp = self.P, self.P["dt"], self.dev, self.dtype
        B = len(imgs81)
        x = torch.tensor(imgs81, device=dev, dtype=dtp)
        Iapp = I_APP * x.transpose(1, 2).reshape(B, -1)
        VE = torch.full((B, self.NE), p["V_rest"], device=dev, dtype=dtp)
        UE = torch.full((B, self.NE), p["U_rest"], device=dev, dtype=dtp)
        VI = torch.full((B, self.NI), p["V_rest"], device=dev, dtype=dtp)
        UI = torch.full((B, self.NI), p["U_rest"], device=dev, dtype=dtp)
        IsynE = torch.zeros_like(VE)
        IsynI = torch.zeros_like(VI)
        count = torch.zeros((B, self.NE), device=dev, dtype=torch.int32)
        ring = torch.zeros((self.n_win, B, self.A), device=dev, dtype=torch.int32)
        timer = torch.zeros((B, self.A), device=dev, dtype=torch.int32)
        modE = torch.zeros_like(VE)
        used = torch.zeros((B, self.A), device=dev, dtype=torch.bool)
        pad = torch.zeros((B, 1), device=dev, dtype=torch.bool)
        vs = p["v_spike"]
        for s in range(self.n_steps):
            fE, fI = VE >= vs, VI >= vs
            VE = torch.where(fE, p["c"], VE); UE = UE + p["d"] * fE
            VI = torch.where(fI, p["c"], VI); UI = UI + p["d"] * fI
            I_in = Iapp if s < self.on_steps else 0.0
            VEn = VE + dt * 1000 * (0.04 * VE * VE + 5 * VE + 140 + I_in + IsynE - UE)
            UE = UE + dt * 1000 * p["a"] * (p["b"] * VEn - UE)
            VIn = VI + dt * 1000 * (0.04 * VI * VI + 5 * VI + 140 + IsynI - UI)
            UI = UI + dt * 1000 * p["a"] * (p["b"] * VIn - UI)
            VE, VI = VEn.clamp(max=vs), VIn.clamp(max=vs)
            spk = VE >= vs
            count += spk
            SE = torch.sigmoid(VE / p["k_syn"])
            SI = torch.sigmoid(VI / p["k_syn"])
            sumEE = torch.sparse.mm(self.W["EE"], SE.T).T
            sumIE = torch.sparse.mm(self.W["IE"], SI.T).T
            sumEI = torch.sparse.mm(self.W["EI"], SE.T).T
            I_EE = (sumEE * (1 + p["nu_ca"] * modE) * (p["E_exc"] - VE)).clamp(min=0)
            IsynE = I_EE + sumIE * (p["E_inh"] - VE)
            IsynI = sumEI * (p["E_exc"] - VI)
            ring[s % self.n_win] = spk[:, self.members].sum(-1)
            if (s + 1) % self.watch_every == 0:
                on = self.ca_ok[None] & (ring.max(0).values >= SYNC_MIN)
                timer = torch.where(on, self.t_astro, timer)
            mod_z = timer > 0
            used |= mod_z
            modE = torch.cat([mod_z, pad], 1)[:, self.neuron_zones].any(-1).to(dtp)
            timer = (timer - 1).clamp(min=0)
        WE = self.P["WE"]
        counts = count.reshape(B, WE, WE).transpose(1, 2)
        return counts.cpu().numpy(), used.cpu().numpy()


def zones_to81(Z):
    N, M, K = Z.shape
    z = int(P["ZONE"])
    st = z - 1
    out = np.zeros((N, st * M + 1, st * K + 1), Z.dtype)
    for j in range(M):
        for k in range(K):
            out[:, st * j:st * j + z, st * k:st * k + z] = Z[:, j, k][:, None, None]
    return out


def used_to_zones(used):
    return used.reshape(len(used), P["NA"], P["MA"]).transpose(0, 2, 1)


def snan_to28(counts, used):
    if SNAN_OUTPUT == "astro":
        return down28(zones_to81(used_to_zones(used).astype(np.uint8)))
    return down28(counts >= SPIKE_THR)


def snan_process(net, B81, label):
    t0 = time.time()
    counts = np.zeros(B81.shape, np.int16)
    used = np.zeros((len(B81), net.A), bool)
    for k in range(0, len(B81), SNAN_BATCH):
        counts[k:k + SNAN_BATCH], used[k:k + SNAN_BATCH] = net.run(B81[k:k + SNAN_BATCH])
    X = snan_to28(counts, used)
    print(f"    {label}: modulated zones {used.sum(1).mean():.1f}, output = 1 for {X.mean() * 100:.1f}% of pixels "
          f"({time.time() - t0:.0f} s)")
    return X, counts, used


def snan_settings(trained):
    return dict(TRAINED_FILE=os.path.basename(trained), TRAINED_MTIME=float(round(os.path.getmtime(trained))),
                OUTPUT=SNAN_OUTPUT, NOISE_TYPE=NOISE_TYPE, DATASET=DATASET, SYNC_MIN=SYNC_MIN,
                SHRINK=float(SHRINK or 0), AUG_SQUARE=int(bool(AUG_SQUARE)))


def epoch_curve_noise():
    return EPOCH_CURVE_NOISE if NOISE_TYPE == "impulse" and TRAIN_NOISE != "matched" else None


def snan_levels():
    ec = epoch_curve_noise()
    test = sorted({round(float(v), 4) for v in LEVELS + ([ec] if ec is not None else [])})
    if TRAIN_NOISE == "matched":
        train_lv = [round(float(v), 4) for v in LEVELS]
    else:
        train_lv = [train_key()]
    return test, train_lv


def feature_file(name, run):
    return os.path.join(FEATURES_DIR, f"{name}_{NOISE_TYPE}_run{run}.mat")


def snan_features_ok(f, trained, ite, itr):
    if not os.path.isfile(f):
        return None
    try:
        D = sio.loadmat(f, squeeze_me=True, struct_as_record=False,
                        variable_names=["settings", "test_index", "train_index", "test_noise_levels", "train_noise_levels"])
        old = {k: getattr(D["settings"], k) for k in D["settings"]._fieldnames}
        new = snan_settings(trained)
        same = all(k in old and (str(old[k]) == str(v) if isinstance(v, str) else np.isclose(float(old[k]), float(v)))
                   for k, v in new.items())
        test, train_lv = snan_levels()
        lt = {round(float(v), 4) for v in np.atleast_1d(D["test_noise_levels"])}
        lr = {round(float(v), 4) for v in np.atleast_1d(D["train_noise_levels"])}
        if not (same and set(test) <= lt and np.array_equal(np.atleast_1d(D["test_index"]), ite)
                and np.array_equal(np.atleast_1d(D["train_index"]), itr)):
            return None
        return "ok" if set(train_lv) <= lr else "train"
    except Exception:
        return None


def save_atomic(out, D):
    tmp = out + ".tmp.mat"
    sio.savemat(tmp, D, do_compression=True)
    os.replace(tmp, out)


def make_snan_features(run, name, net, trained, out, Btr, Bte, raw_tr, raw_te, itr, ite, only_train):
    seed = 42 + run - 1
    test, train_lv = snan_levels()
    if only_train:
        D = {k: v for k, v in sio.loadmat(out).items() if not k.startswith("__")}
        have = [round(float(v), 4) for v in np.asarray(D.get("train_noise_levels", np.zeros(0))).reshape(-1)]
        new_lv = [nu for nu in train_lv if nu not in have]
        print(f"  [run {run}] {name}: training images {new_lv}")
        parts = {nu: D["train_X"][have.index(nu)] for nu in have}
        for nu in new_lv:
            parts[nu] = snan_process(net, make_train(Btr, raw_tr, nu, seed), f"train {nu:g}")[0]
        allv = sorted(parts)
        D["train_X"] = np.stack([parts[nu] for nu in allv]).astype(np.float32)
        D["train_noise_levels"] = np.array(allv, float)
        save_atomic(out, D)
        return
    print(f"  [run {run}] {name}: test images")
    test_X, ex_in, ex_cnt, ex_astro = [], [], [], []
    for nu in test:
        Bn = corrupt_test(Bte, raw_te, nu, seed)
        X, counts, used = snan_process(net, Bn, f"noise {nu:g}")
        test_X.append(X)
        ex_in.append(Bn[:N_EXAMPLES]); ex_cnt.append(counts[:N_EXAMPLES])
        ex_astro.append(zones_to81(used_to_zones(used[:N_EXAMPLES]).astype(np.uint8)))
    D = dict(test_X=np.stack(test_X).astype(np.float32), test_index=ite, test_noise_levels=np.array(test, float),
             train_index=itr, train_noise_levels=np.array(train_lv, float), settings=snan_settings(trained),
             example_input81=np.stack(ex_in), example_counts81=np.stack(ex_cnt), example_astro81=np.stack(ex_astro))
    print(f"  [run {run}] {name}: training images")
    D["train_X"] = np.stack([snan_process(net, make_train(Btr, raw_tr, nu, seed), f"train {nu:g}")[0]
                             for nu in train_lv]).astype(np.float32)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_atomic(out, D)
    snan_plot_examples(D, os.path.splitext(out)[0] + ".png")


def snan_plot_examples(D, fig_file, idx=0):
    from matplotlib.figure import Figure
    nus = D["test_noise_levels"]
    pick = (0, 25, 50, 75) if NOISE_TYPE == "occlusion" else (0.0, 0.15, 0.3, 0.5)
    show = [i for i, v in enumerate(nus) if any(np.isclose(v, q) for q in pick)] or [0]
    fig = Figure(figsize=(3.2 * len(show), 12.5))
    ax = fig.subplots(4, len(show), squeeze=False)
    for j, i in enumerate(show):
        ax[0, j].imshow(D["example_input81"][i, idx], cmap="gray"); ax[0, j].set_title(f"input, noise {nus[i]:g}")
        ax[1, j].imshow(D["example_counts81"][i, idx], cmap="hot"); ax[1, j].set_title(f"spikes in {TEST_WINDOW_MS:g} ms")
        ax[2, j].imshow(D["example_astro81"][i, idx], cmap="gray", vmin=0, vmax=1); ax[2, j].set_title("modulated zones")
        ax[3, j].imshow(D["test_X"][i, idx], cmap="gray", vmin=0, vmax=1); ax[3, j].set_title(f"SNAN output ({SNAN_OUTPUT})")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(fig_file, dpi=110)


def prepare_snan_features(mnist, dev, names=("SNAN", "SB_SNAN")):
    t0 = time.time()
    Xtr_all, Ytr_all, Xte_all, Yte_all = mnist
    _, snan_file, sb_file = NETS[NOISE_TYPE]
    for name, trained in [("SNAN", snan_file), ("SB_SNAN", sb_file)]:
        if name not in names:
            continue
        if not os.path.isfile(trained):
            raise FileNotFoundError(f"{trained} not found")
        net = None
        for run in range(1, NUM_RUNS + 1):
            out = feature_file(name, run)
            itr, ite = pick_split(Ytr_all, Yte_all, N_TRAIN_PER_CLASS, N_TEST_PER_CLASS, 42 + run - 1)
            status = snan_features_ok(out, trained, ite, itr)
            if status == "ok":
                continue
            if net is None:
                Pf, W, Ca = load_snan(trained, dev, TEST_DTYPE)
                net = SNANTest(Pf, W, Ca, dev, TEST_DTYPE)
                print(f"{name}: {trained}, {int((Ca >= TEST_CA_THR).sum())} of {Ca.size} astrocytes with Ca >= {TEST_CA_THR}")
            raw_tr, raw_te = shrink28(Xtr_all[itr]), Xte_all[ite]
            Btr, Bte = binarize81(raw_tr, N_THREADS_DATA), binarize81(shrink28(raw_te), N_THREADS_DATA)
            make_snan_features(run, name, net, trained, out, Btr, Bte, raw_tr, raw_te, itr, ite,
                               only_train=(status == "train"))
        del net
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"SNAN outputs ready ({time.time() - t0:.0f} s)\n")


def load_snan_features(name, run, ite, itr, dev):
    D = sio.loadmat(feature_file(name, run))
    if not np.array_equal(np.asarray(D["test_index"]).reshape(-1), ite):
        raise ValueError("test images differ")
    if not np.array_equal(np.asarray(D["train_index"]).reshape(-1), itr):
        raise ValueError("training images differ")
    lv = [round(float(v), 4) for v in np.asarray(D["test_noise_levels"]).reshape(-1)]
    test, train_lv = snan_levels()
    tset = {nu: torch.tensor(D["test_X"][lv.index(round(nu, 4))][:, None].astype(np.float32), device=dev) for nu in test}
    tl = [round(float(v), 4) for v in np.asarray(D["train_noise_levels"]).reshape(-1)]
    trset = {nu: torch.tensor(D["train_X"][tl.index(nu)][:, None].astype(np.float32)) for nu in train_lv}
    return tset, trset


def window_sum(B, before, after, mode):
    Pd = np.pad(B.astype(np.int32), ((0, 0), (before, after), (before, after)), mode=mode)
    c = np.zeros((Pd.shape[0], Pd.shape[1] + 1, Pd.shape[2] + 1), np.int32)
    c[:, 1:, 1:] = Pd.cumsum(1).cumsum(2)
    w = before + after + 1
    H, W = B.shape[1], B.shape[2]
    return c[:, w:w + H, w:w + W] - c[:, :H, w:w + W] - c[:, w:w + H, :W] + c[:, :H, :W]


def apply_filter(B, name, dev=None):
    if name == "mean":
        k = FILTER_P["mean_size"]
        s = window_sum(B, (k - 1) // 2, k // 2, "symmetric")
        return (2 * s >= k * k).astype(np.uint8)
    if name == "median":
        k = FILTER_P["median_size"]
        s = window_sum(B, (k - 1) // 2, k // 2, "constant")
        need = (k * k) // 2 + 1 if k % 2 else (k * k) // 2
        return (s >= need).astype(np.uint8)
    sig, degree = FILTER_P["bilat_sigma"], FILTER_P["bilat_degree"]
    dev = dev or torch.device("cpu")
    r = int(np.ceil(2 * sig))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    ks = np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))
    e = np.exp(-1.0 / (2 * degree))
    Bt = torch.tensor(B, dtype=torch.float64, device=dev)[:, None]
    k = torch.tensor(ks, dtype=torch.float64, device=dev)[None, None]
    S1 = F.conv2d(F.pad(Bt, (r, r, r, r), mode="replicate"), k)[:, 0]
    S0 = float(ks.sum())
    one = Bt[:, 0] == 1
    num = torch.where(one, S1, e * S1)
    den = torch.where(one, S1 + e * (S0 - S1), (S0 - S1) + e * S1)
    return (num / den >= 0.5).to(torch.uint8).cpu().numpy()


class CNN(nn.Module):
    def __init__(self, mean_image, N1=200, N2=100, N3=10, size=IMG_SIZE):
        super().__init__()
        self.register_buffer("mean", mean_image.reshape(1, 1, size, size).float())
        self.conv_1 = nn.Conv2d(1, 8, 3, padding=1);  self.BN_1 = nn.BatchNorm2d(8)
        self.conv_2 = nn.Conv2d(8, 16, 3, padding=1); self.BN_2 = nn.BatchNorm2d(16)
        self.conv_3 = nn.Conv2d(16, 32, 3, padding=1); self.BN_3 = nn.BatchNorm2d(32)
        self.FC_buffer = nn.Linear((size // 4) * (size // 4) * 32, N1)
        self.FC_attack = nn.Linear(N1, N2)
        self.FC_1 = nn.Linear(N2, N3)

    def embed(self, x):
        x = 255 * x - self.mean
        x = F.max_pool2d(F.relu(self.BN_1(self.conv_1(x))), 2, 2)
        x = F.max_pool2d(F.relu(self.BN_2(self.conv_2(x))), 2, 2)
        x = F.relu(self.BN_3(self.conv_3(x)))
        x = x.permute(0, 1, 3, 2).flatten(1)
        x = F.relu(self.FC_buffer(x))
        return F.relu(self.FC_attack(x))

    def head(self, e):
        return self.FC_1(e)

    def forward(self, x):
        return self.head(self.embed(x))


def glorot_init(model, gen):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            fan_in = m.weight[0].numel()
            fan_out = m.weight.shape[0] * m.weight[0, 0].numel()
            b = np.sqrt(6.0 / (fan_in + fan_out))
            with torch.no_grad():
                m.weight.uniform_(-b, b, generator=gen)
            nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.out = nn.Linear(heads * dim_head, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, mlp)
        self.ff2 = nn.Linear(mlp, dim)

    def forward(self, z):
        B, T, _ = z.shape
        q, k, v = self.qkv(self.norm1(z)).view(B, T, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        z = z + self.out(a.transpose(1, 2).reshape(B, T, -1))
        return z + self.ff2(F.relu(self.ff1(self.norm2(z))))


class ViT(nn.Module):
    def __init__(self, patch=4, dim=128, depth=8, heads=8, dim_head=64, mlp=256, n_cls=10, gen=None, size=IMG_SIZE):
        super().__init__()
        self.p = patch
        n = (size // patch) ** 2
        self.to_patch = nn.Linear(patch * patch, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim, generator=gen))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim, generator=gen))
        self.layers = nn.ModuleList([Block(dim, heads, dim_head, mlp) for _ in range(depth)])
        self.head_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, n_cls)

    def embed(self, x):
        B = x.shape[0]
        p = self.p
        x = x.unfold(2, p, p).unfold(3, p, p).reshape(B, -1, p * p)
        z = torch.cat([self.cls.expand(B, -1, -1), self.to_patch(x)], 1) + self.pos
        for blk in self.layers:
            z = blk(z)
        return self.head_norm(z[:, 0])

    def head(self, e):
        return self.fc(e)

    def forward(self, x):
        return self.head(self.embed(x))


def kaiming_default_init(model, gen):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            b = 1.0 / np.sqrt(m.weight.shape[1])
            with torch.no_grad():
                m.weight.uniform_(-b, b, generator=gen)
                if m.bias is not None:
                    m.bias.uniform_(-b, b, generator=gen)


def random_translate(x, max_shift, gen):
    B, _, H, W = x.shape
    sh = (torch.rand(B, 2, generator=gen, device=x.device) * 2 - 1) * max_shift
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2 * sh[:, 0] / W
    theta[:, 1, 2] = -2 * sh[:, 1] / H
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


@torch.no_grad()
def embed_all(model, X, bs=None, allowed=None):
    bs = bs or (500 if X.shape[-1] <= 32 else 50)
    model.eval()
    e = torch.cat([model.embed(X[k:k + bs]) for k in range(0, len(X), bs)])
    logits = model.head(e)
    if allowed is not None:
        mask = torch.full((logits.shape[1],), float("-inf"), device=logits.device)
        mask[list(allowed)] = 0
        logits = logits + mask
    pred = logits.argmax(1)
    model.train()
    return e.double().cpu().numpy(), pred.cpu().numpy()


def cluster_metrics(emb, pred, labels, cent0=None, sf0=None):
    cls = np.unique(labels)
    cent = np.full((10, emb.shape[1]), np.nan)
    size = np.full(10, np.nan)
    for c in cls:
        cent[c] = emb[labels == c].mean(0)
        size[c] = np.mean(np.linalg.norm(emb[labels == c] - cent[c], axis=1))
    dist = np.linalg.norm(cent[:, None] - cent[None], axis=2)
    sf = np.sqrt(np.trace(np.cov(emb, rowvar=False)))
    sub = dist[np.ix_(cls, cls)]
    off = ~np.eye(len(cls), dtype=bool)
    return dict(
        accuracy=float(np.mean(pred == labels)),
        Cs=float(np.mean(size[cls] / sf)),
        Cd=float(np.mean(sub[off] / sf)),
        minCd=float(np.min(sub[off]) / sf),
        shift=(float(np.mean(np.linalg.norm(cent[cls] - cent0[cls], axis=1)) / sf0) if cent0 is not None else 0.0),
        centroids=cent,
        sf=float(sf),
    )


def train_model(name, Xtr, Ytr, dev, seed, epoch_eval=None, tag="", over=None):
    gen_cpu = torch.Generator().manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    size = Xtr.shape[-1]
    if name == "CNN":
        Pm = {**CNN_P, **(over or {})}
        with torch.no_grad():
            mean_image = torch.cat([random_translate(Xtr[k:k + 512], Pm["augment_shift"], gen)
                                    for k in range(0, len(Xtr), 512)]).mean(0) * 255
        model = CNN(mean_image.cpu(), Pm["N1"], Pm["N2"], size=size)
        glorot_init(model, gen_cpu)
    else:
        Pm = {**VIT_P, **(over or {})}
        model = ViT(Pm["patch"], Pm["dim"], Pm["depth"], Pm["heads"], Pm["dim_head"], Pm["mlp"], gen=gen_cpu, size=size)
        kaiming_default_init(model, gen_cpu)
    model.to(dev).train()
    params = list(model.parameters())
    if name == "CNN":
        vel = [torch.zeros_like(p) for p in params]
    else:
        opt = torch.optim.Adam(params, lr=Pm["lr"])
    N, bs = len(Xtr), Pm["batch"]
    acc_curve, losses, it = [], [], 0
    allowed = epoch_eval[2] if epoch_eval is not None and len(epoch_eval) > 2 else None
    if epoch_eval is not None:
        acc_curve.append(float((embed_all(model, epoch_eval[0], allowed=allowed)[1] == epoch_eval[1]).mean()))
    for epoch in range(1, Pm["epochs"] + 1):
        order = torch.randperm(N, generator=gen, device=dev)
        ep_losses = []
        for k in range(0, N, bs):
            idx = order[k:k + bs]
            x, y = Xtr[idx], Ytr[idx]
            if name == "CNN":
                x = random_translate(x, Pm["augment_shift"], gen)
            loss = F.cross_entropy(model(x), y)
            it += 1
            if name == "CNN":
                grads = torch.autograd.grad(loss, params)
                lr = Pm["lr0"] / (1 + Pm["decay"] * it)
                with torch.no_grad():
                    for p, g, v in zip(params, grads, vel):
                        v.mul_(Pm["momentum"]).sub_(lr * g)
                        p.add_(v)
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            ep_losses.append(loss.detach())
        losses.extend(torch.stack(ep_losses).cpu().tolist())
        if epoch_eval is not None:
            acc_curve.append(float((embed_all(model, epoch_eval[0], allowed=allowed)[1] == epoch_eval[1]).mean()))
        if epoch % 10 == 0 or epoch == Pm["epochs"]:
            print(f"{tag}{name} epoch {epoch}, loss {losses[-1]:.4f}")
    return model, acc_curve, losses


KEYS = ["accuracy", "Cs", "Cd", "minCd", "shift", "centroids", "sf"]
METRICS = ["accuracy", "Cs", "Cd", "minCd", "shift"]


def one_run(run, mnist, dev):
    t0 = time.time()
    tag = f"[run {run}] "
    seed = 42 + run - 1
    Xtr_all, Ytr_all, Xte_all, Yte_all = mnist
    itr, ite = pick_split(Ytr_all, Yte_all, N_TRAIN_PER_CLASS, N_TEST_PER_CLASS, seed)
    nthr = max(1, N_THREADS_DATA // N_PARALLEL_RUNS)
    raw_tr, raw_te = shrink28(Xtr_all[itr]), Xte_all[ite]
    Btr, Bte = binarize81(raw_tr, nthr), binarize81(shrink28(raw_te), nthr)
    Ytr = torch.tensor(Ytr_all[itr], device=dev)
    Yte = Yte_all[ite]
    ec = epoch_curve_noise()
    lv_all = LEVELS + ([ec] if ec is not None and ec not in LEVELS else [])
    noisy81 = {nu: corrupt_test(Bte, raw_te, nu, seed) for nu in lv_all}
    test = {nu: to_tensor(down28(B), dev) for nu, B in noisy81.items()}

    class Filtered(dict):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def __getitem__(self, nu):
            return to_tensor(down28(apply_filter(noisy81[nu], self.f, dev)), dev)

    ftest = {f: Filtered(f) for f in FILTERS}

    def train_set(nu):
        return to_tensor(down28(make_train(Btr, raw_tr, nu, seed)), dev)

    snan_test, snan_train = load_snan_features("SNAN", run, ite, itr, dev)
    sb_test, sb_train = load_snan_features("SB_SNAN", run, ite, itr, dev)
    sb_sel_np = np.flatnonzero(np.isin(Yte, SB_CLASSES))
    sb_sel = torch.tensor(sb_sel_np, device=dev)
    Yte_sb = Yte[sb_sel_np]

    def evaluate(model, X, labels, allowed=None, cent0=None, sf0=None):
        return cluster_metrics(*embed_all(model, X, allowed=allowed), labels, cent0, sf0)

    def noise_curve(model, tset, labels, sel=None, allowed=None):
        res = {k: [] for k in KEYS}
        cent0 = sf0 = None
        for nu in LEVELS:
            X = tset[nu] if sel is None else tset[nu][sel]
            m = evaluate(model, X, labels, allowed, cent0, sf0)
            if cent0 is None:
                cent0, sf0 = m["centroids"], m["sf"]
            for k in res:
                res[k].append(m[k])
        return res

    def add(dst, m):
        for k in KEYS:
            dst[k].append(m[k])

    out = {}
    for mi, name in enumerate(CLASSIFIERS):
        mseed = seed * 10 + mi
        filters = FILTERS if name == "CNN" else []
        variants = {"snan": (snan_test, None, Yte, None), "snantt": (snan_test, snan_train, Yte, None),
                    "sbsnan": (sb_test, None, Yte_sb, sb_sel), "sbsnantt": (sb_test, sb_train, Yte_sb, sb_sel)}
        vseed = {"snantt": 5, "sbsnantt": 7}
        if TRAIN_NOISE == "matched":
            res = {k: [] for k in KEYS}
            fres = {f: {k: [] for k in KEYS} for f in filters}
            vres = {v: {k: [] for k in KEYS} for v in variants}
            for nu in LEVELS:
                key = round(float(nu), 4)
                model, _, _ = train_model(name, train_set(nu), Ytr, dev, mseed, tag=f"{tag}[train noise {nu}] ")
                add(res, evaluate(model, test[nu], Yte))
                for f in filters:
                    add(fres[f], evaluate(model, ftest[f][nu], Yte))
                for v, (tset, trset, labels, sel) in variants.items():
                    allowed = SB_CLASSES if sel is not None else None
                    X = tset[nu] if sel is None else tset[nu][sel]
                    if trset is None:
                        add(vres[v], evaluate(model, X, labels, allowed))
                    else:
                        m2, _, _ = train_model(name, trset[key].to(dev), Ytr, dev, mseed + vseed[v],
                                               tag=f"{tag}[{v}, noise {nu}] ")
                        add(vres[v], evaluate(m2, X, labels, allowed))
                        del m2
            for r in [res, *fres.values(), *vres.values()]:
                r["shift"] = [np.nan] * len(LEVELS)
            curve, losses = [], []
            for f in filters:
                out[f"{name}_{f}"] = {**fres[f], "epoch_accuracy": [], "loss": []}
            for v in variants:
                out[f"{name}_{v}"] = {**vres[v], "epoch_accuracy": [], "loss": []}
        else:
            ev = (test[ec], Yte) if ec is not None else None
            model, curve, losses = train_model(name, train_set(train_key()), Ytr, dev, mseed, epoch_eval=ev, tag=tag)
            res = noise_curve(model, test, Yte)
            for f in filters:
                out[f"{name}_{f}"] = {**noise_curve(model, ftest[f], Yte), "epoch_accuracy": [], "loss": []}
            for v, (tset, trset, labels, sel) in variants.items():
                allowed = SB_CLASSES if sel is not None else None
                if trset is None:
                    out[f"{name}_{v}"] = {**noise_curve(model, tset, labels, sel, allowed), "epoch_accuracy": [], "loss": []}
                else:
                    ev2 = None
                    if ec is not None:
                        ev2 = (tset[ec], Yte) if sel is None else (tset[ec][sel], labels, allowed)
                    m2, c2, l2 = train_model(name, trset[train_key()].to(dev), Ytr, dev, mseed + vseed[v],
                                             epoch_eval=ev2, tag=f"{tag}[{v}] ")
                    out[f"{name}_{v}"] = {**noise_curve(m2, tset, labels, sel, allowed), "epoch_accuracy": c2, "loss": l2}
                    del m2
        out[name] = {**res, "epoch_accuracy": curve, "loss": losses}
        for m in [name] + [f"{name}_{f}" for f in filters] + [f"{name}_{v}" for v in variants]:
            print(f"{tag}{m}: " + ", ".join(f"{nu:g}: {a:.3f}" for nu, a in zip(LEVELS, out[m]["accuracy"])))
    print(f"{tag}finished in {time.time() - t0:.0f} s")
    return out


def pack(results):
    L, R = len(LEVELS), NUM_RUNS
    D = {"noise_levels": np.array(LEVELS, float), "noise_type": NOISE_TYPE, "train_noise": str(TRAIN_NOISE),
         "aug_min": float(AUG_MIN or 0), "snan_output": SNAN_OUTPUT, "runs_done": np.zeros(R), "dataset": DATASET,
         "snan_size": SNAN_SIZE, "sync_min": SYNC_MIN, "shrink": float(SHRINK or 0), "aug_square": int(bool(AUG_SQUARE))}
    names = []
    for res in results.values():
        names += [n for n in res if n not in names]
    for name in names:
        base = name.split("_")[0]
        PP = CNN_P if base == "CNN" else VIT_P
        ep = PP["epochs"]
        it = ep * int(np.ceil(10 * N_TRAIN_PER_CLASS / PP["batch"]))
        dim = CNN_P["N2"] if base == "CNN" else VIT_P["dim"]
        A = {k: np.full((L, R), np.nan) for k in METRICS + ["sf"]}
        A["centroids"] = np.full((10, dim, L, R), np.nan)
        A["epoch_accuracy"] = np.full((ep + 1, R), np.nan)
        A["loss"] = np.full((it, R), np.nan)
        for r, res in results.items():
            x = res[name]
            for k in METRICS + ["sf"]:
                A[k][:, r] = x[k]
            A["centroids"][:, :, :, r] = np.stack(x["centroids"], 2)
            if x["epoch_accuracy"]:
                A["epoch_accuracy"][:, r] = x["epoch_accuracy"]
            if x["loss"]:
                A["loss"][:len(x["loss"]), r] = x["loss"]
        done = [r for r in results]
        for k in METRICS:
            v = A[k][:, done]
            A[f"{k}_mean"] = np.nanmean(v, 1)
            A[f"{k}_sd"] = np.nanstd(v, 1, ddof=1) if len(done) > 1 else np.zeros(L)
        for k, v in A.items():
            D[f"{name}_{k}"] = v
    for r in results:
        D["runs_done"][r] = 1
    return D


COLORS = {"CNN": "#1f77b4", "ViT": "#7b3fa0",
          "CNN_mean": "#ff7f0e", "CNN_median": "#2ca02c", "CNN_bilateral": "#d62728",
          "CNN_snan": "#e6b800", "CNN_snantt": "#b8860b", "CNN_sbsnan": "#8fbf3f", "CNN_sbsnantt": "#4a7f1f",
          "ViT_snan": "#f08cc0", "ViT_snantt": "#c2185b", "ViT_sbsnan": "#9fa8da", "ViT_sbsnantt": "#3949ab"}
LABELS = {"CNN": "CNN", "ViT": "ViT",
          "CNN_mean": "Mean filter + CNN", "CNN_median": "Median filter + CNN", "CNN_bilateral": "Bilateral filter + CNN",
          "CNN_snan": "SNAN (test) + CNN", "CNN_snantt": "SNAN (train and test) + CNN",
          "ViT_snan": "SNAN (test) + ViT", "ViT_snantt": "SNAN (train and test) + ViT",
          "CNN_sbsnan": "S-B SNAN (test) + CNN *", "CNN_sbsnantt": "S-B SNAN (train and test) + CNN *",
          "ViT_sbsnan": "S-B SNAN (test) + ViT *", "ViT_sbsnantt": "S-B SNAN (train and test) + ViT *"}
ORDER = ["CNN", "CNN_mean", "CNN_median", "CNN_bilateral", "CNN_snan", "CNN_snantt", "CNN_sbsnan", "CNN_sbsnantt",
         "ViT", "ViT_snan", "ViT_snantt", "ViT_sbsnan", "ViT_sbsnantt"]


def save_summary(D, fname):
    nu = D["noise_levels"].reshape(-1)
    models = [m for m in ORDER if f"{m}_accuracy_mean" in D]
    with open(fname, "w", encoding="utf-8") as f:
        f.write("model;noise;" + ";".join(f"{k} mean;{k} sd" for k in METRICS) + "\n")
        for m in models:
            for i, v in enumerate(nu):
                vals = []
                for k in METRICS:
                    vals += [D[f"{m}_{k}_mean"].reshape(-1)[i], D[f"{m}_{k}_sd"].reshape(-1)[i]]
                f.write(f"{m};{v:g};" + ";".join("" if np.isnan(x) else f"{x:.4f}" for x in vals) + "\n")


def plot(fname):
    import warnings
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    D = sio.loadmat(fname)
    base = os.path.splitext(fname)[0]
    nu = D["noise_levels"].reshape(-1)
    occl = str(D["noise_type"][0]) == "occlusion"
    if occl:
        nu = nu / SNAN_SIZE * 100
    models = [m for m in ORDER if f"{m}_accuracy_mean" in D]
    panels = [("accuracy", "Accuracy"), ("Cs", r"$\bar C_s$"), ("Cd", r"$\bar C_d$"),
              ("minCd", r"min $\bar C_d$"), ("shift", "Cluster shift")]
    if np.all(np.isnan(np.concatenate([D[f"{m}_shift_mean"].ravel() for m in models]))):
        panels = panels[:4]
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for j, (key, lab) in enumerate(panels):
        for m in models:
            mu, sd = D[f"{m}_{key}_mean"].reshape(-1), D[f"{m}_{key}_sd"].reshape(-1)
            ax[j].plot(nu, mu, "o-", color=COLORS[m], label=LABELS[m], ms=4, lw=1.5)
            ax[j].fill_between(nu, mu - sd, mu + sd, color=COLORS[m], alpha=0.25, lw=0)
        ax[j].set_xlabel("Occlusion size (% of image side)" if occl else "Test noise level")
        ax[j].set_ylabel(lab)
        ax[j].grid(True, alpha=0.4)
    ax[0].legend(fontsize=8)
    runs = int(D["runs_done"].sum())
    if occl:
        title = (f"Binarized {DATASET}, occlusion test; training augmentation: random scale "
                 f"{int(D['aug_min'].ravel()[0])}-79 px of {SNAN_SIZE} + shift, {runs} runs (mean ± SD)")
    else:
        title = f"Binarized {DATASET}, train noise: {D['train_noise'][0]}, {runs} runs (mean ± SD)"
    title += f"\n* S-B SNAN: test images of classes {SB_CLASSES} only, prediction among these classes"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.show()
    curve_models = [m for m in models if np.any(~np.isnan(D[f"{m}_epoch_accuracy"]))]
    if curve_models:
        fig2, a2 = plt.subplots(figsize=(5, 3.6))
        done = D["runs_done"].reshape(-1) > 0
        for m in curve_models:
            a = D[f"{m}_epoch_accuracy"][:, done]
            ep = np.arange(a.shape[0])
            mu = np.nanmean(a, 1)
            sd = np.nanstd(a, 1, ddof=1) if a.shape[1] > 1 else 0 * mu
            a2.plot(ep, mu, color=COLORS[m], label=LABELS[m], lw=1.5)
            a2.fill_between(ep, mu - sd, mu + sd, color=COLORS[m], alpha=0.25, lw=0)
        a2.set_xlabel("Epoch")
        a2.set_ylabel(f"Accuracy (test noise {EPOCH_CURVE_NOISE})")
        a2.grid(True, alpha=0.4)
        a2.legend(fontsize=7)
        fig2.tight_layout()
        fig2.savefig(base + "_epochs.png", dpi=200)
        plt.show()
    save_summary(D, base + "_summary.csv")
    print("Mean ± SD over runs:")
    for m in models:
        print(m)
        for i, v in enumerate(D["noise_levels"].reshape(-1)):
            print(f"  {v:g}: " + "  ".join(
                f"{k} {D[f'{m}_{k}_mean'].reshape(-1)[i]:.3f}±{D[f'{m}_{k}_sd'].reshape(-1)[i]:.3f}" for k in METRICS))


def run_experiment(name, noise="impulse", train_noise=0.0, aug_min=None):
    global NOISE_TYPE, TRAIN_NOISE, AUG_MIN, LEVELS, SHRINK
    import pickle
    NOISE_TYPE = noise
    TRAIN_NOISE = train_noise if noise == "impulse" else 0.0
    AUG_MIN = aug_min if noise == "occlusion" else None
    SHRINK = OCCLUSION_SHRINK if noise == "occlusion" else None
    LEVELS = list(OCCLUSION_LEVELS if noise == "occlusion" else IMPULSE_LEVELS)
    out_file = os.path.join(RESULTS_DIR, name + ".mat")
    run_dir = os.path.join(RESULTS_DIR, name + "_runs")
    os.makedirs(run_dir, exist_ok=True)
    run_file = lambda r: os.path.join(run_dir, f"run{r}.pkl")
    results = {}
    for r in range(1, NUM_RUNS + 1):
        if os.path.isfile(run_file(r)):
            with open(run_file(r), "rb") as f:
                results[r - 1] = pickle.load(f)
    todo = [r for r in range(1, NUM_RUNS + 1) if r - 1 not in results]
    if not todo:
        if not os.path.isfile(out_file):
            save_atomic(out_file, pack(results))
        print(f"{out_file}: all {NUM_RUNS} runs already done")
        plot(out_file)
        return
    if results:
        print(f"{name}: runs {sorted(r + 1 for r in results)} already done, computing runs {todo}")
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = USE_TF32
        torch.backends.cudnn.allow_tf32 = USE_TF32
    mnist = load_mnist(MNIST_ROOT)
    prepare_snan_features(mnist, dev)
    lock = threading.Lock()

    def job(run):
        res = one_run(run, mnist, dev)
        with lock:
            results[run - 1] = res
            tmp = run_file(run) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(res, f)
            os.replace(tmp, run_file(run))
            save_atomic(out_file, pack(results))

    with ThreadPoolExecutor(N_PARALLEL_RUNS) as ex:
        for f in [ex.submit(job, r) for r in todo]:
            f.result()
    print(f"Saved {out_file}, total time {time.time() - t0:.0f} s")
    plot(out_file)


CURVE_MODELS = ["CNN", "CNN_snantt", "ViT", "ViT_snantt"]


def run_jobs(name, keys, job, finish):
    import pickle
    run_dir = os.path.join(RESULTS_DIR, name + "_runs")
    os.makedirs(run_dir, exist_ok=True)
    run_file = lambda k: os.path.join(run_dir, f"n{k[0]}_run{k[1]}.pkl")
    results = {}
    for k in keys:
        if os.path.isfile(run_file(k)):
            with open(run_file(k), "rb") as f:
                results[k] = pickle.load(f)
    todo = [k for k in keys if k not in results]
    if results:
        print(f"{name}: {len(results)} of {len(keys)} jobs already done")
    lock = threading.Lock()

    def one(k):
        res = job(*k)
        with lock:
            results[k] = res
            tmp = run_file(k) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(res, f)
            os.replace(tmp, run_file(k))
            finish(results)

    with ThreadPoolExecutor(N_PARALLEL_RUNS) as ex:
        for f in [ex.submit(one, k) for k in todo]:
            f.result()
    finish(results)


def pack_curves(results, sizes, epochs, extra):
    R = NUM_RUNS
    D = {"sizes": np.array(sizes, float), "epochs": np.arange(epochs + 1), **extra}
    for m in CURVE_MODELS:
        for n in sizes:
            A = np.full((epochs + 1, R), np.nan)
            for (nn_, r), res in results.items():
                if nn_ == n and m in res:
                    A[:, r - 1] = res[m]
            done = ~np.all(np.isnan(A), 0)
            D[f"{m}_n{n}_accuracy"] = A
            D[f"{m}_n{n}_accuracy_mean"] = np.nanmean(A[:, done], 1) if done.any() else np.full(epochs + 1, np.nan)
            D[f"{m}_n{n}_accuracy_sd"] = np.nanstd(A[:, done], 1, ddof=1) if done.sum() > 1 else np.zeros(epochs + 1)
            D[f"{m}_n{n}_runs_done"] = int(done.sum())
    return D


def plot_curves(fname, title, size_label):
    import warnings
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    D = sio.loadmat(fname)
    base = os.path.splitext(fname)[0]
    sizes = [int(v) for v in D["sizes"].ravel()]
    ep = D["epochs"].ravel()
    fig, ax = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.8), sharey=True, squeeze=False)
    for j, n in enumerate(sizes):
        a = ax[0, j]
        for m in CURVE_MODELS:
            mu, sd = D[f"{m}_n{n}_accuracy_mean"].ravel(), D[f"{m}_n{n}_accuracy_sd"].ravel()
            a.plot(ep, mu, color=COLORS[m], label=LABELS[m], lw=1.5)
            a.fill_between(ep, mu - sd, mu + sd, color=COLORS[m], alpha=0.25, lw=0)
        a.set_title(f"{n} {size_label}")
        a.set_xlabel("Epoch")
        a.grid(True, alpha=0.4)
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.show()
    fig2, a2 = plt.subplots(figsize=(5, 3.8))
    for m in CURVE_MODELS:
        mu = np.array([D[f"{m}_n{n}_accuracy_mean"].ravel()[-1] for n in sizes])
        sd = np.array([D[f"{m}_n{n}_accuracy_sd"].ravel()[-1] for n in sizes])
        a2.errorbar(sizes, mu, sd, color=COLORS[m], label=LABELS[m], marker="o", capsize=3)
    a2.set_xscale("log")
    a2.set_xlabel(size_label)
    a2.set_ylabel(f"Accuracy after {int(ep[-1])} epochs")
    a2.grid(True, alpha=0.4)
    a2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(base + "_final.png", dpi=200)
    plt.show()
    with open(base + "_summary.csv", "w", encoding="utf-8") as f:
        f.write("model;" + size_label + ";epoch;accuracy mean;accuracy sd\n")
        for m in CURVE_MODELS:
            for n in sizes:
                mu, sd = D[f"{m}_n{n}_accuracy_mean"].ravel(), D[f"{m}_n{n}_accuracy_sd"].ravel()
                for e in ep:
                    f.write(f"{m};{n};{e};{mu[e]:.4f};{sd[e]:.4f}\n")
    print(f"Accuracy after {int(ep[-1])} epochs, mean ± SD:")
    for m in CURVE_MODELS:
        print(f"  {m}: " + "  ".join(f"{n}: {D[f'{m}_n{n}_accuracy_mean'].ravel()[-1]:.3f}±{D[f'{m}_n{n}_accuracy_sd'].ravel()[-1]:.3f}"
                                     for n in sizes))


def setup_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = USE_TF32
        torch.backends.cudnn.allow_tf32 = USE_TF32
    return dev


def run_small_samples(name="small_samples"):
    global NOISE_TYPE, TRAIN_NOISE, AUG_MIN, LEVELS, SHRINK
    NOISE_TYPE, TRAIN_NOISE, AUG_MIN, SHRINK = "impulse", 0.0, None, None
    LEVELS = list(IMPULSE_LEVELS)
    out_file = os.path.join(RESULTS_DIR, name + ".mat")
    dev = setup_device()
    mnist = load_mnist(MNIST_ROOT)
    prepare_snan_features(mnist, dev, names=("SNAN",))
    Xtr_all, Ytr_all, Xte_all, Yte_all = mnist
    over = {"epochs": SMALL_EPOCHS}

    def job(n, run):
        tag = f"[n {n}, run {run}] "
        seed = 42 + run - 1
        itr, ite = pick_split(Ytr_all, Yte_all, N_TRAIN_PER_CLASS, N_TEST_PER_CLASS, seed)
        sel = np.arange(len(itr)).reshape(10, N_TRAIN_PER_CLASS)[:, :n].ravel()
        nthr = max(1, N_THREADS_DATA // N_PARALLEL_RUNS)
        Btr = binarize81(Xtr_all[itr[sel]], nthr)
        Bte = binarize81(Xte_all[ite], nthr)
        Ytr = torch.tensor(Ytr_all[itr[sel]], device=dev)
        Yte = Yte_all[ite]
        Xte = to_tensor(down28(corrupt_test(Bte, Xte_all[ite], SMALL_NOISE, seed)), dev)
        snan_test, snan_train = load_snan_features("SNAN", run, ite, itr, dev)
        Str = snan_train[train_key()][torch.as_tensor(sel)].to(dev)
        Ste = snan_test[round(SMALL_NOISE, 4)]
        res = {}
        for mi, m in enumerate(CLASSIFIERS):
            mseed = seed * 10 + mi
            _, c, _ = train_model(m, to_tensor(down28(Btr), dev), Ytr, dev, mseed, epoch_eval=(Xte, Yte), tag=tag, over=over)
            res[m] = c
            _, c, _ = train_model(m, Str, Ytr, dev, mseed + 5, epoch_eval=(Ste, Yte), tag=tag + "[snantt] ", over=over)
            res[f"{m}_snantt"] = c
        print(tag + "  ".join(f"{k} {v[-1]:.3f}" for k, v in res.items()))
        return res

    def finish(results):
        save_atomic(out_file, pack_curves(results, SMALL_SIZES, SMALL_EPOCHS,
                                          {"dataset": DATASET, "test_noise": SMALL_NOISE}))

    keys = [(n, r) for n in SMALL_SIZES for r in range(1, NUM_RUNS + 1)]
    run_jobs(name, keys, job, finish)
    plot_curves(out_file, f"Binarized {DATASET}: training on n images per class (clean), test noise {SMALL_NOISE}, "
                          f"{NUM_RUNS} runs (mean ± SD)", "images per class")


def train_snan_omniglot():
    train(OMNI_SNAN_FILE, "MNIST", geom=OMNI_GEOM)


def omniglot_index(root):
    from torchvision.datasets import Omniglot
    ds = Omniglot(root, background=True, download=True)
    files = [[] for _ in ds._characters]
    for img, c in ds._flat_character_images:
        files[c].append(os.path.join(ds.target_folder, ds._characters[c], img))
    return files


def omniglot_split(files, split):
    from PIL import Image
    rng = np.random.default_rng(42 + split - 1)
    chars = rng.choice(len(files), OMNI_CLASSES, replace=False)
    imgs, labels = [], []
    for lab, c in enumerate(chars):
        fs = sorted(files[c])
        for p in rng.permutation(len(fs)):
            imgs.append(255 - np.asarray(Image.open(fs[p]).convert("L"), np.uint8))
            labels.append(lab)
    return np.stack(imgs), np.array(labels)


def omniglot_pad(B):
    a = (OMNI_INPUT - B.shape[1]) // 2
    b = OMNI_INPUT - B.shape[1] - a
    return np.pad(B, ((0, 0), (a, b), (a, b))).astype(np.float32)


def omniglot_features(files, dev):
    saved = dict(P)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    net = None
    try:
        for split in range(1, NUM_RUNS + 1):
            out = os.path.join(FEATURES_DIR, f"omniglot_run{split}.mat")
            st = dict(TRAINED_FILE=os.path.basename(OMNI_SNAN_FILE),
                      TRAINED_MTIME=float(round(os.path.getmtime(OMNI_SNAN_FILE))), SYNC_MIN=SYNC_MIN, NOISE=OMNI_NOISE)
            if os.path.isfile(out):
                try:
                    old = sio.loadmat(out, squeeze_me=True, struct_as_record=False, variable_names=["settings"])["settings"]
                    if all(np.isclose(float(getattr(old, k)), float(v)) if not isinstance(v, str) else str(getattr(old, k)) == v
                           for k, v in st.items()):
                        continue
                except Exception:
                    pass
            if net is None:
                Pf, W, Ca = load_snan(OMNI_SNAN_FILE, dev, TEST_DTYPE)
                net = SNANTest(Pf, W, Ca, dev, TEST_DTYPE)
                print(f"Omniglot SNAN: {OMNI_SNAN_FILE}, {int((Ca >= TEST_CA_THR).sum())} of {Ca.size} astrocytes with Ca >= {TEST_CA_THR}")
            imgs, labels = omniglot_split(files, split)
            B = binarize81(imgs, N_THREADS_DATA, size=P["WE"])
            Bn = add_noise(B, OMNI_NOISE, np.random.default_rng([42 + split - 1, 2, int(round(OMNI_NOISE * 1000))]))
            print(f"  [split {split}] Omniglot: {len(B)} images")
            Xc = snan_process(net, B, "clean")[2]
            Xn = snan_process(net, Bn, f"noise {OMNI_NOISE:g}")[2]
            Z = lambda u: zones_to81(used_to_zones(u).astype(np.uint8))
            save_atomic(out, dict(B=B, Bn=Bn, S=Z(Xc), Sn=Z(Xn), labels=labels, settings=st))
    finally:
        P.clear()
        P.update(saved)
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def run_omniglot(name="omniglot"):
    out_file = os.path.join(RESULTS_DIR, name + ".mat")
    if not os.path.isfile(OMNI_SNAN_FILE):
        raise FileNotFoundError(f"{OMNI_SNAN_FILE} not found")
    dev = setup_device()
    files = omniglot_index(OMNIGLOT_ROOT)
    omniglot_features(files, dev)
    over = {"CNN": {"epochs": OMNI_EPOCHS}, "ViT": {"epochs": OMNI_EPOCHS, "patch": OMNI_PATCH}}

    def job(k, split):
        tag = f"[{k}-shot, split {split}] "
        seed = 42 + split - 1
        D = sio.loadmat(os.path.join(FEATURES_DIR, f"omniglot_run{split}.mat"))
        labels = D["labels"].ravel()
        per = len(labels) // OMNI_CLASSES
        pos = np.arange(len(labels)).reshape(OMNI_CLASSES, per)
        tr, te = pos[:, :k].ravel(), pos[:, k:].ravel()
        T = lambda X, idx: torch.tensor(omniglot_pad(X[idx])[:, None], device=dev)
        Ytr = torch.tensor(labels[tr].astype(np.int64), device=dev)
        Yte = labels[te].astype(np.int64)
        res = {}
        for mi, m in enumerate(CLASSIFIERS):
            mseed = seed * 10 + mi
            _, c, _ = train_model(m, T(D["B"], tr), Ytr, dev, mseed, epoch_eval=(T(D["Bn"], te), Yte), tag=tag, over=over[m])
            res[m] = c
            _, c, _ = train_model(m, T(D["S"], tr), Ytr, dev, mseed + 5, epoch_eval=(T(D["Sn"], te), Yte),
                                  tag=tag + "[snantt] ", over=over[m])
            res[f"{m}_snantt"] = c
        print(tag + "  ".join(f"{key} {v[-1]:.3f}" for key, v in res.items()))
        return res

    def finish(results):
        save_atomic(out_file, pack_curves(results, OMNI_SHOTS, OMNI_EPOCHS, {"dataset": "Omniglot", "test_noise": OMNI_NOISE}))

    keys = [(k, s) for k in OMNI_SHOTS for s in range(1, NUM_RUNS + 1)]
    run_jobs(name, keys, job, finish)
    plot_curves(out_file, f"Omniglot, {OMNI_CLASSES} random characters, k training images per class (clean), "
                          f"test on the rest, noise {OMNI_NOISE}, {NUM_RUNS} splits (mean ± SD)", "images per class")
