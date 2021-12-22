# AIDA
AIDA is an **A**ctive **I**nference-based **D**esign **A**gent that aims at real-time situated client-driven design of audio processing algoritms

## Get data
We use DVC tool to store/track the dataset that's used in this project.
To obtain the datasets please run the following commands.
```bash
cd demo/verification

dvc get git@github.com:biaslab/AIDA-data.git verification-jlds
```
You can skip this if you prefer to run the experiments from scratch.

To obtain the dataset for the validation experiments, please run in the project main folder:
```
dvc get git@github.com:biaslab/AIDA-data.git sound
```
## Run experiments
To run the experiments, execute the following commands in **Julia REPL**

1. <code>] activate .</code>
2. <code>] instantiate</code>

### AIDA folders structure
```
├─ demo
│  ├─ validation
│  └─ verification
│     ├─ verification-jlds
|     ├─ tmp
|     ├─ tikz
├─ sound
│  ├─ separated_jld
│  │  └─ speech
│  │     ├─ babble
│  │     │  └─ 5dB
│  │     ├─ sin
│  │     └─ train
│  │        └─ 5dB
│  └─ speech
│     ├─ babble
│     │  ├─ 0dB
│     │  └─ 5dB
│     ├─ clean
│     ├─ sin
│     └─ train
│        ├─ 0dB
│        └─ 5dB
├─ src
│  ├─ agent
│  ├─ application
│  ├─ environment
│  ├─ helpers

```
