#!/usr/bin/python3
# See https://github.com/pyannote/pyannote-audio
import sys
USAGE = "Usage: pyanote.py inputfile outputfile HuggingFace-access-token"

if len(sys.argv) < 1: sys.exit(USAGE)
if len(sys.argv) > 1: infile=sys.argv[1]
if len(sys.argv) > 2: outfile=sys.argv[2]
if len(sys.argv) > 3: HUGGINGFACE_ACCESS_TOKEN=sys.argv[3]
if len(sys.argv) > 4: sys.exit(str(len(sys.argv)-1) + " args! " + USAGE)

from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_ACCESS_TOKEN)

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline(infile)

# print the result
maxspeakers = 0
with open(outfile, 'w', encoding="utf-8") as out:
  print("start end speaker", file=out) # headerw
  for turn, _, speaker in diarization.itertracks(yield_label=True):
    s = int(turn.start * 1000)
    e = int(turn.end * 1000)
    w = int(speaker[8:])        # speaker is "SPEAKER_NN", we just take the int
    if w > maxspeakers: maxspeakers = w
    print(f'{s} {e} {w}', file=out)
print(f'== Detected {maxspeakers+1} different speakers')
