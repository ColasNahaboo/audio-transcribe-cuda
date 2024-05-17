# audio-transcribe-cuda
These are wrapper scripts in python and bash to transcribe audio to text via whisper and other open source libs via a Nvidia card. I made these for my personal needs, but are publishing them as they may be useful to others.

The problem:
- I recorded some confcalls (via a phone in speaker mode and another phone recording the call) that I wanted to transcribe to text to have minutes of the call.
- I have an old linux spare PC with an old NVDIA graphic card, not powerful enough for modern 3D games, but with decent memory (6G) and able to run GPU "cuda" code: a Nvidia RTX 1060 with 5G RAM (what is called the "Pascal" architecture. This code will work with any more recent NVidia card with at **least 6G VRAM.**

This solution:
- use the whisper, ffmpeg, pyannote libraries via their python APIs/wrappers
- but actually use faster-whisper to be able to use the best model, `large-v3` in a 6G GPU, as the normal whisper will not fit, and French transcription is not very good with the smaller models.
- bash scripts to glue things together. This is because I am a bash expert but a novice in python. I may try to recode these in python in the future, but no promises.
- It also provides option to clean and equalize voices in a confcall, and detect the different speakers (aka diarisation).

This way I can transcribe a recording at twice the speed of it, i.e. transcribe a 20mn recording in less than 10mn on my mere RTX 1060

## Installation:
- Decide on a directory where this system will reside on the linux machine with the Nvidia card. For instance `/opt/audio-transcribe`
- **Download** there this repository: 
  `cd /opt && git clone git@github.com:ColasNahaboo/audio-transcribe.git`
  Note that if you cannot write as a user in `/opt`, you can create `/opt/audio-transcribe` as root and make it read+writable by you, and then use https://github.com/tj/git-extras/blob/master/man/git-force-clone.md to download into it.
  If you installed it into `/opt/audio-transcribe`, you may want to create symlinks to the scripts in a directory in your path, e.g:
  `ln -sf /opt/{audio-transcribe-cuda,convert-transcribe-cuda} /usr/local/bin`
- **Install python3, ffmpeg, mediainfo** if they are not already installed.
  On Debian: `sudo apt -y install python3 ffmpeg mediainfo`
- **Install faster-whisper**,  follow its instructions at https://github.com/SYSTRAN/faster-whisper and do not forget to install the GPU libs via `pip install nvidia-cublas-cu11 nvidia-cudnn-cu11`
  On Debian, if pip install fails, you may have to force the install by: `sudo pip install nvidia-cublas-cu11 nvidia-cudnn-cu11 --break-system-packages`
- **Install pyanote-audio**, follow the instructions at https://github.com/pyannote/pyannote-audio if you want to identify the different speakers in the recording, what is called "speaker diarisation".
  You will have to create an access token on Hugging Face and store this token in the file `HUGGINGFACE_ACCESS_TOKEN.txt` in the directory
- If pyannote seems very slow, see the fix in: https://github.com/pyannote/pyannote-audio/issues/1481#issuecomment-1741039112
  `pip uninstall onnxruntime; pip install --force-reinstall onnxruntime-gpu`
- Optionally, install `torchvision` to supress a warning message from pyannote, but it is not needed: `pip install torchvision`
- No need to install the actual whisper ( https://github.com/openai/whisper ) as fast-whisper replaces it.

## Configuration
Configuration declarations can be made in two files that are read as bash scripts, so variable declarations in them should be in bash syntax:
- `/etc/default/audio-transcribe-cuda` system-wide, for all users
- `~/.config/audio-transcribe-cuda` per user
The list of definable options can be found in the section `Options default values` in the bash script `audio-transcribe-cuda` itself.


## History
- v0.1.0 2024-05-17  Alpha version. Works but lacks features.
