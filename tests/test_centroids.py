import pytest
import os
import matplotlib.pyplot as plt

import lightkurve as lk
from vetting import centroid_test

lib_path = os.getcwd()
tpf = lk.KeplerTargetPixelFile(f"{lib_path}/tests/data/kplr005562784-2011271113734_lpd-targ.fits.gz")

def test_centroid_test():
    period, t0, dur = 25.3368592, 192.91552, 8.85/24
    r = centroid_test(tpf, period, t0, dur, aperture_mask='pipeline', plot=False)
    assert (r['pvalues'][0][0] < 0.01)
    assert len(r['pvalues'][0]) == 1

    r = centroid_test(tpf, period, t0, dur, aperture_mask='pipeline', plot=True)
    assert len(r['figs']) == 1
    r['figs'][0].savefig(f'{lib_path}/demo.png', dpi=200, bbox_inches='tight')

    r = centroid_test(tpf, [period, 22.4359873459], [t0, 0], [dur, dur], aperture_mask='pipeline', plot=False)
    # True transit should be significant
    assert (r['pvalues'][0][0] < 0.01)
    # Random transit shouldn't be significant
    assert (r['pvalues'][0][1] > 0.5)
    assert len(r['pvalues'][0]) == 2

    r = centroid_test([tpf, tpf], [period, 22.4359873459], [t0, 0], [dur, dur], aperture_mask='pipeline', plot=False)
    assert (len(r['pvalues']) == 2)
    assert (len(r['pvalues'][0]) == 2)
