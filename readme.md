# Running steps

- To download and process data, run:

```
python preprocessdata.py
```

- Then, we train our model:

```
python train.py
```

- To see how to use our model, see `predict.py`, with input is the string to the video, and out is either 0 or 1.

```
from utils import file2class
result = file2class('pat/to/file/name')
```
