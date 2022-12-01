import numpy as np
import vugrad_modified as vg
from vugrad_modified.vugrad.ops import Normalize

if __name__ == '__main__':
    a = vg.TensorNode(np.random.randn(2, 2))
    b = vg.TensorNode(np.random.randn(2, 2))
    c = a + b
    n_forward = Normalize.do_forward(a)
    Normalize.backward(n_forward.source.context, n_forward.value)
    n_forward.backward()
    c.source
    pass
