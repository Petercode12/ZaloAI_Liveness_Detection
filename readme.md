# Problem Statement
In verification services related to face recognition (such as eKYC and face access control), the key question is whether the input face video is real (from a live person present at the point of capture), or fake (from a spoof artifact or lifeless body). Liveness detection is the AI problem to answer that question.

# Running steps

- To download and process data, run:

```
python preprocessdata.py
```

- Then, we train our model:

```
python train.py
```

- To run our model:

```
python predict.py
```
