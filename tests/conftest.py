import numpy as np
import pytest


def synthetic_underwater(h=96, w=128, seed=0, cast=(0.25, 0.75, 0.55), haze=0.35):
    """A textured scene degraded with a colour cast and a veil."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    scene = np.dstack(
        [
            0.5 + 0.4 * np.sin(xx / 7.0) * np.cos(yy / 11.0),
            0.5 + 0.4 * np.sin(xx / 13.0 + 1.0),
            0.5 + 0.4 * np.cos(yy / 9.0),
        ]
    )
    scene[h // 3 : h // 2, w // 3 : w // 2] = 0.9
    scene += rng.normal(0, 0.01, scene.shape)
    img = (1 - haze) * scene * np.array(cast) + haze * np.array(cast)
    return np.clip(img, 0, 1).astype(np.float32)


@pytest.fixture
def greenish():
    return synthetic_underwater()


@pytest.fixture
def bluish():
    return synthetic_underwater(cast=(0.2, 0.5, 0.8))
