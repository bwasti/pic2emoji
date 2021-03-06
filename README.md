# pic2emoji

For some context, read the [write-up](https://jott.live/markdown/pic2emoji).

![](https://i.imgur.com/wFFZ3ib.png)

### Download
```
git clone https://github.com/bwasti/pic2emoji.git
pip install tinygrad
```

You can optionally install `pyopencl` for more performance during training time.

### Train
```
python train.py emoji_set
```
You can add more emoji to the folder.
They'll need to be 72x72.
A good resource is this site: https://emojipedia.org/apple/

### Infer

Note: the size of the image must be divisible by 36

```
python train.py emoji_set mario.png
```

This will generate an image `mario.png.emoji.png`.

### Contributions

Happy to take PRs!
However, this is a weekend project so I won't be actively maintaining it much.
