import os


os.system("python main.py --arch mlp --nepoch 70")
os.system("python main.py --arch attn --nepoch 50")
os.system("python main.py --arch attn --sample_weight --nepoch 50")
os.system("python main.py --arch gann --nepoch 80")
os.system("python main.py --arch gann --sample_weight  --nepoch 50")

