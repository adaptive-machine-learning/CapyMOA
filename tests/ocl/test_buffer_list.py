import pytest

pytestmark = pytest.markskip("torch")

import torch  # noqa: E402

from capymoa.ocl.util._buffer_list import BufferList  # noqa: E402


def test_buffer_list_mutations_preserve_order_and_identity():
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    c = torch.tensor([3.0])
    d = torch.tensor([4.0])

    buffers = BufferList([a, c])

    assert len(buffers) == 2
    assert buffers[0] is a
    assert buffers[1] is c
    assert a in buffers
    assert torch.tensor([1.0]) not in buffers

    buffers.insert(1, b)

    assert len(buffers) == 3
    assert buffers[0] is a
    assert buffers[1] is b
    assert buffers[2] is c

    del buffers[1]

    assert len(buffers) == 2
    assert buffers[0] is a
    assert buffers[1] is c
    assert list(buffers._buffers.keys()) == ["0", "1"]

    buffers[1] = d

    assert buffers[1] is d
    assert c not in buffers
