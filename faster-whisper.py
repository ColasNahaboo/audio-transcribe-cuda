#!/usr/bin/python3
# See https://github.com/SYSTRAN/faster-whisper
import sys
USAGE = "Usage: faster-whisper.py inputfile outputfile.tsv model best_of beam_size computation"

outfile = "out.tsv"
model = "large-v3"
beamsize = 5
#compute =  "float16" # our 1060GT does not support float16
#compute =  "float32" # faster-whisper does not support
computation = "int8_float16"
language = ""

if len(sys.argv) < 1: sys.exit(USAGE)
if len(sys.argv) > 1: infile=sys.argv[1]
if len(sys.argv) > 2: outfile=sys.argv[2]
if len(sys.argv) > 3: model=sys.argv[3]
if len(sys.argv) > 4: beamsize=int(sys.argv[4])
if len(sys.argv) > 5: computation=sys.argv[5]
if len(sys.argv) > 6: language=sys.argv[6]
if len(sys.argv) > 7: sys.exit(str(len(sys.argv)-1) + " args! " + USAGE)

# Run whisper on GPU
from faster_whisper import WhisperModel
model = WhisperModel(model, device="cuda", compute_type=computation)

# export the transcribed sgments into a TSV file
if language:
  segments, info = model.transcribe(infile, beam_size=beamsize, task="translate", language=language)
else:
  segments, info = model.transcribe(infile, beam_size=beamsize)

# print the file
with open(outfile, 'w', encoding="utf-8") as out:
  print("start end text", file=out)         # header
  for segment in segments:
    s = int(segment.start * 1000)
    e = int(segment.end * 1000)
    t = segment.text
    print(f'{s} {e} {t}', file=out)
