# Non-IID Image Classifiation

Class Project for DIP(Digital Image Processing) @ THU-CST

Implemented 2-layer-MLP, Attention-Mask-MLP as baseline. we use our modified [DANN](https://dl.acm.org/doi/abs/10.5555/2946645.2946704)(use GAN method instead of Gradient Reversal, and 3-stage training instead of cooperative training) as our proposed method, which also support [RFF Sampling Weight method](https://arxiv.org/abs/2104.07876).

run:

```bash
cd code
python main.py -h
// some runinng examples:
python main.py --arch mlp --nepoch 70
python main.py --arch attn --nepoch 50
python main.py --arch attn --sample_weight --nepoch 5
python main.py --arch gann --nepoch 80
python main.py --arch gann --sample_weight  --nepoch 50
```

