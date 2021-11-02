import os

import lightkurve as lk
import matplotlib.pyplot as plt
import pytest
import numpy as np

from vetting import centroid_test
from vetting import PACKAGEDIR

testdir = "/".join(PACKAGEDIR.split("/")[:-2])


def is_action():
    try:
        return os.environ["GITHUB_ACTIONS"]
    except KeyError:
        return False


def test_centroid_test():
    tpf = lk.KeplerTargetPixelFile(
        f"{testdir}/tests/data/kplr005562784-2011271113734_lpd-targ.fits.gz"
    )
    period, t0, dur = 25.3368592, 192.91552, 8.85 / 24
    r = centroid_test(tpf, period, t0, dur, aperture_mask="pipeline", plot=False)
    assert r["pvalues"][0][0] < 0.01
    assert len(r["pvalues"][0]) == 1

    r = centroid_test(tpf, period, t0, dur, aperture_mask="pipeline", plot=True)
    assert len(r["figs"]) == 1
    r["figs"][0].savefig(f"{PACKAGEDIR}/demo.png", dpi=200, bbox_inches="tight")

    r = centroid_test(
        tpf,
        [period, 22.4359873459],
        [t0, 0],
        [dur, dur],
        aperture_mask="pipeline",
        plot=False,
    )
    # True transit should be significant
    assert r["pvalues"][0][0] < 0.01
    # Random transit shouldn't be significant
    assert r["pvalues"][0][1] > 0.2
    assert len(r["pvalues"][0]) == 2

    r = centroid_test(
        [tpf, tpf],
        [period, 22.4359873459],
        [t0, 0],
        [dur, dur],
        aperture_mask="pipeline",
        plot=False,
    )
    assert len(r["pvalues"]) == 2
    assert len(r["pvalues"][0]) == 2

    r = centroid_test(
        tpf,
        period,
        t0,
        dur,
        aperture_mask="pipeline",
        plot=True,
        transit_depths=0.001499,
    )
    assert "1sigma_error" in r.keys()
    assert hasattr(r["1sigma_error"][0], "unit")
    assert r["1sigma_error"][0].value > 0
    assert r["1sigma_error"][0].value < 0.5

    r = centroid_test(
        tpf,
        period,
        t0,
        dur,
        aperture_mask="pipeline",
        plot=True,
        transit_depths=0.001499,
        labels="c",
    )


@pytest.mark.skipif(
    is_action(),
    reason="Can not run on GitHub actions, because it's a pain to download.",
)
def test_FPs():
    """Produce some figures that show centroid offsets"""
    names = [
        "KIC 5391911",
        "KIC 5866724",
        "EPIC 220256496",
        "EPIC 210957318",
        "TIC 293435336",
        "TIC 13023738",
    ]
    kwargs = [
        {"quarter": 3},
        {"quarter": 3},
        {"campaign": 8},
        {"campaign": 4},
        {},
        {"sector": 2},
    ]

    periods = [0.782068488, 2.154911236, 0.669558, 4.098503000, 1.809886, 8.0635]
    t0s = [
        131.8940201,
        133.498613,
        2457393.81364 - 2454833,
        2457063.80710000 - 2454833,
        2456107.85507 - 2457000,
        2459104.87232 - 2457000,
    ]
    durs = np.asarray([1.239, 3.1341, 1, 2.3208, 3.694, 3.396]) / 24
    depths = [
        0.01157 * 0.01,
        0.00850 * 0.01,
        0.015450 ** 2,
        1.603 * 0.01,
        1.189 * 0.01,
        5180 * 1e-6,
    ]

    for name, period, t0, dur, kwarg, depth in zip(
        names, periods, t0s, durs, kwargs, depths
    ):
        tpf = lk.search_targetpixelfile(name, **kwarg).download()
        r = centroid_test(
            tpf,
            period,
            t0,
            dur,
            aperture_mask="pipeline",
            plot=True,
            transit_depths=depth,
        )
        r["figs"][0].savefig(
            f"{testdir}/docs/{name.replace(' ','').lower()}.png",
            dpi=200,
            bbox_inches="tight",
        )
        # nb = int(1 // np.median(np.diff(tpf.time.value)))
        # nb = [nb if nb % 2 == 1 else nb + 1][0]
        # ax = tpf.to_lightcurve().flatten(nb).fold(period, t0).bin(period / 200).plot()
        # plt.savefig(
        #     f"{testdir}/docs/{name.replace(' ','').lower()}_lc.png",
        #     dpi=200,
        #     bbox_inches="tight",
        # )
