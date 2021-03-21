from PIL import Image
import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.optim as optimizer

import os
import sys

if len(sys.argv) < 2:
    print(f"usage: python {sys.argv[0]} dataset_folder")
    exit(1)


def blur(a, k):
    kernel = np.ones(k, dtype=np.float32)
    a = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 2, a)
    a = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 3, a)
    a /= k ** 2
    return a


def color(a, k, pct):
    a = np.copy(a)
    a[:, k, :, :] *= 1 + pct
    a = a.clip(0, 255)
    return a


# from https://stackoverflow.com/questions/27087139/shifting-an-image-in-numpy
def shift(a, ox, oy):
    non = lambda s: s if s < 0 else None
    mom = lambda s: max(0, s)

    out = np.zeros_like(a)
    out[:, :, mom(oy) : non(oy), mom(ox) : non(ox)] = a[
        :, :, mom(-oy) : non(-oy), mom(-ox) : non(-ox)
    ]
    return out


def perturb(d):
    # blur radius no more than 14% of image width
    b = np.random.randint(0, 5)
    if b:
        d = blur(d, b)

    c = np.random.randint(0, 4)
    # change color no more than 14%
    pct = np.random.rand(1)[0] / 7 * np.random.randint(-1, 2)
    if c and pct:  # 1,2,3
        d = color(d, c - 1, pct)

    # shift no more than 14% each direction
    sx = np.random.randint(-5, 5)
    sy = np.random.randint(-5, 5)
    d = shift(d, sx, sy)
    return d


def small(data):
    return data[:, ::2, ::2]


def load_to_np(f):
    data = np.asarray(Image.open(f))
    data = data.transpose(2, 0, 1)
    data = data.astype(np.float32)
    data = small(data)
    return data


files = [np.zeros((4, 36, 36), dtype=np.float32)]
filenames = os.listdir(sys.argv[1])
for f in sorted(filenames):
    if not f.endswith("png"):
        continue
    try:
        d = load_to_np(os.path.join(sys.argv[1], f))
        if len(d.shape) != 3:
            continue
        if d.shape[0] != 4:
            continue
        files.append(d)
    except:
        continue
files = np.array(files)


gpu = True
try:
    import pyopencl

    print("using GPU to train")
except:
    print("no GPU, falling back to CPU.  perhaps install pyopencl?")
    gpu = False


def to_gpu(t):
    if gpu:
        return t.gpu()
    return t


# from here: https://github.com/geohot/tinygrad/blob/master/examples/train_efficientnet.py#L13
class TinyConvNet:
    def __init__(self, classes, saved=None):
        conv = 3
        inter_chan, out_chan = 8, 16
        if saved:
            self.c1 = Tensor(saved["arr_0"])
            self.c2 = Tensor(saved["arr_1"])
            self.l1 = Tensor(saved["arr_2"])
        else:
            self.c1 = to_gpu(Tensor.uniform(inter_chan, 4, conv, conv))
            self.c2 = to_gpu(Tensor.uniform(out_chan, inter_chan, conv, conv))
            self.l1 = to_gpu(Tensor.uniform(out_chan * 7 * 7, classes))

    # I want to eventually run this in the browser
    # real time
    #
    # ~2.2MFlops, wasm hits ~1GFlops so ~500/sec
    # but we have batch size ~5k for high res inputs
    # so we need (at 15 inf/sec) a 13KFlops model
    # 99.5% sparse? ...hmm
    def forward(self, x):
        # 3 * 8 * 2 * 3 * 3 * 36 * 36 -> 0.6M
        x = x.conv2d(self.c1).relu().max_pool2d()
        # 8 * 16 * 2 * 9 * 18 * 18 -> 0.8M
        x = x.conv2d(self.c2).relu().max_pool2d()
        x = x.reshape(shape=[x.shape[0], -1])
        # 49 * 16 * 1024 -> 0.8M
        return x.dot(self.l1).logsoftmax()


def train(model):
    optim = optimizer.SGD([model.c1, model.c2, model.l1], lr=0.001)
    iters = 5 * classes
    BS = 32
    log_skip = 5
    print(f"training with {classes} emojis over {BS * iters} iters batched by {BS}")
    import time

    t0 = time.time()
    for _ in range(iters):
        samp = np.random.randint(0, files.shape[0], size=(BS))
        img = to_gpu(Tensor(perturb(files[samp])))
        out = model.forward(img)
        y = np.zeros((BS, classes), dtype=np.float32)
        # cross entropy loss trick
        y[range(y.shape[0]), samp] = -classes
        y = to_gpu(Tensor(y))
        loss = out.mul(y).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if _ % log_skip == 0:
            t1 = time.time()
            iters_sec = log_skip * BS / (t1 - t0)
            minutes = (BS * (iters - _) / iters_sec) / 60
            sys.stdout.write("\u001b[1000D")
            sys.stdout.write("\u001b[0K")
            sys.stdout.write(
                f"[{100 * _ / iters:.2f}% ~{int(minutes)}min left] loss={loss.cpu().data[0]:.2f}"
            )
            sys.stdout.write(
                f" (running @ {iters_sec:.2f} iters/s, feel free to kill at any time)"
            )
            sys.stdout.flush()
        t0 = t1


def infer(model, img):
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    img = Tensor(img)
    out = np.argmax(model.forward(img).data, axis=1)
    return out


classes = len(files)
if len(sys.argv) == 3:
    o = np.load(sys.argv[1] + ".npz")
    model = TinyConvNet(classes, o)
    data = np.asarray(Image.open(sys.argv[2]))
    data = data.transpose(2, 0, 1)
    data = data.astype(np.float32)
    max_size = 36
    data = np.array(np.array_split(data, data.shape[2] // max_size, axis=2))
    data = np.array(np.array_split(data, data.shape[2] // max_size, axis=2))
    for h in range(data.shape[0]):
        sys.stdout.write("\u001b[1000D")
        sys.stdout.write("\u001b[0K")
        sys.stdout.write(f"{100 * h / data.shape[0]:.2f}%")
        sys.stdout.flush()
        idx = infer(model, data[h])
        data[h, :, :, :, :] = files[idx]
    data = data.transpose(0, 3, 1, 4, 2)
    data = data.reshape(
        data.shape[0] * data.shape[1], data.shape[2] * data.shape[3], data.shape[4]
    )
    data = data.astype(np.uint8)
    f = sys.argv[2] + ".emoji.png"
    Image.fromarray(data).save(f)
    print(f"\ndone. saved to {f}")
else:
    model = TinyConvNet(classes)
    try:
        train(model)
    except KeyboardInterrupt:
        pass
    print("\nsaving model...")
    np.savez(
        sys.argv[1] + ".npz",
        model.c1.cpu().data,
        model.c2.cpu().data,
        model.l1.cpu().data,
    )
    print(f"infer with: python {' '.join(sys.argv[0:2])} your_img.png")
