import numpy as np
import sys

sys.path.insert(0, ".")

from src.robotics.models.HistoryBuffer import HistoryBuffer


def _image():
    return {"cam": np.zeros((2, 2, 3), dtype=np.uint8)}


def _state(t):
    return np.array([t], dtype=np.float32)


def test_prev_actions_match_training_alignment():
    buf = HistoryBuffer(history_length=3, action_dim=2)
    a0 = np.array([1.0, 10.0], dtype=np.float32)
    a1 = np.array([2.0, 20.0], dtype=np.float32)

    buf.push(_image(), _state(0))
    np.testing.assert_allclose(buf.get_batch()["prev_actions"].numpy()[0], np.zeros((3, 2)))

    buf.record_action(a0)
    buf.push(_image(), _state(1))
    np.testing.assert_allclose(
        buf.get_batch()["prev_actions"].numpy()[0],
        np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 10.0]], dtype=np.float32),
    )

    buf.record_action(a1)
    buf.push(_image(), _state(2))
    np.testing.assert_allclose(
        buf.get_batch()["prev_actions"].numpy()[0],
        np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
    )
