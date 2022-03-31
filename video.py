"""
Create video of a puzzle being built using the sequence
"""
from glob import glob
import os
import subprocess

import numpy as np
from matplotlib import pyplot as plt
from faunadb import query as q
from faunadb.client import FaunaClient

client = FaunaClient(secret=os.environ.get("FAUNA_SECRET"))

sample = client.query(q.get(q.ref(q.collection("jigsawData"), "326505626942308945")))

sequence = sample["data"]["history"]

pieces_per_side = int(np.sqrt(len(sequence)))

jigsaw = np.zeros((pieces_per_side, pieces_per_side))

plt.imshow(jigsaw, vmin=0, vmax=1)
plt.savefig("images/video/00.png")
plt.close()

for i, piece in enumerate(sequence):
    x = piece % pieces_per_side
    y = piece // pieces_per_side

    jigsaw[y, x] = 1

    plt.imshow(jigsaw, vmin=0, vmax=1)
    plt.savefig(f"images/video/{i + 1:02d}.png")
    plt.close()

os.chdir("images/video")
subprocess.call(
    [
        "ffmpeg",
        "-framerate",
        "8",
        "-i",
        "%02d.png",
        "-r",
        "30",
        "-pix_fmt",
        "yuv420p",
        "test.mp4",
    ]
)

for file_name in glob("*.png"):
    os.remove(file_name)
