
### NPL Replication

- Replicating Neural Probabilistic Language Model by Yoshua Bengio(https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

### Usage
```python
from model import NPL, train

model = train("text.txt")
model.generate("this is the", length=10)
```

- train() has configurable parameters given by
```python
    defaults = {
        'hidden_units': 100,
        'context_size': 3,
        'feature_vector_size': 10,
        'direct_rep': False,
        'epochs': 50,
    }

```
- learning rate and batch sizes are not yet configurable but can be edited directly into source code, checkout [edit lr](https://github.com/sayyss/NPL-Replication/blob/main/model.py?plain=1#L152) and [edit batch size](https://github.com/sayyss/NPL-Replication/blob/main/model.py?plain=1#L157) for control.
