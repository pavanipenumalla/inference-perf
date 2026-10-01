# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gaps between session starts on a trace_session_replay stage.

``session_arrival: constant`` spaces starts exactly 1/session_rate apart. ``poisson``
draws each gap from an exponential distribution with that mean, seeded so a run's
arrival times repeat under the same base_seed.
"""

from itertools import islice
from typing import List

import numpy as np

from inference_perf.config import SessionArrival
from inference_perf.loadgen.load_timer import session_start_gaps


def _gaps(rate: float, arrival: SessionArrival, seed: List[int], n: int) -> List[float]:
    return list(islice(session_start_gaps(rate, arrival, seed), n))


def test_constant_gaps_are_one_over_rate() -> None:
    assert _gaps(4.0, SessionArrival.CONSTANT, [42, 0], 5) == [0.25] * 5


def test_poisson_gaps_vary_and_average_one_over_rate() -> None:
    gaps = np.array(_gaps(0.5, SessionArrival.POISSON, [42, 0], 20000))

    assert len(set(gaps.tolist())) > 1
    assert (gaps >= 0).all()
    # Exponential with mean 2s: the sample mean is within 3% at this size, and the
    # coefficient of variation is ~1 (a fixed gap would give 0).
    assert abs(gaps.mean() - 2.0) < 0.06
    assert abs(gaps.std() / gaps.mean() - 1.0) < 0.05


def test_poisson_gaps_repeat_under_the_same_seed() -> None:
    assert _gaps(0.5, SessionArrival.POISSON, [42, 0], 100) == _gaps(0.5, SessionArrival.POISSON, [42, 0], 100)


def test_poisson_gaps_differ_across_seeds_and_stages() -> None:
    base = _gaps(0.5, SessionArrival.POISSON, [42, 0], 100)
    assert base != _gaps(0.5, SessionArrival.POISSON, [43, 0], 100)
    assert base != _gaps(0.5, SessionArrival.POISSON, [42, 1], 100)
