import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel

from scipy.stats import ttest_ind
import corner
import lightkurve as lk

def _label(tpf):
    if hasattr(tpf, 'quarter'):
        return f'{tpf.to_lightcurve().label}, Quarter {tpf.quarter}'
    elif hasattr(tpf, 'campaign'):
        return f'{tpf.to_lightcurve().label}, Campaign {tpf.campaign}'
    elif hasattr(tpf, 'sector'):
        return f'{tpf.to_lightcurve().label}, Sector {tpf.sector}'
    else:
        return '{tpf.to_lightcurve().label}'

def centroid_test(tpfs, periods, t0s, durs, kernel=21, aperture_mask='pipeline', plot=True):
    """
    Runs a simple centroiding test on TPFs for input period, t0 and durations of transiting planet candidates.

    This program will run the following steps for every TPF and planet input:
        1. Find the centroid position as a function of time. Will only use pixels within the pipeline mask
            NOTE: This method will only work if there is not significant crowding inside the pipeline aperture.
            If there is significant crowding, the centroid will shift during any brightness changes.
        2. Remove long term trends in the centroid position using a simple Gaussian smooth (1D Gaussian Kernel Convolution)
        3. Create two distributions of centroids; cadences where there are no planets, and cadences where a given planet transits.
        4. Perform a student t-test to find if the means of the two distributions are consistent, assuming they have the same variance.
            If the p-value of this test is significant (in this case, pvalue < 0.05 or 5%) then the means of the two distributions
            are significantly different, and a centroid offset is detected.
            This test uses the scipy.stats.ttest_ind function.

    Parameters
    ----------
        tpfs : list of lk.TargetPixelFile
            List of target pixel files to perform the centroid test on
        periods : list, np.ndarray or tuple of floats
            The periods at which to perform the test. Should be a list of transiting planet periods present in the TPFs.
        t0s : list, np.ndarray or tuple of floats
            The transit midpoints at which to perform the test. Should be a list of transiting planet transit mid points present in the TPFs.
                NOTE: Make sure the T0s are in the same time baseline as the TPFs
        durs : list, np.ndarray or tuple of floats
            The durations at which to perform the test. Should be a list of transiting planet durations present in the TPFs.
        kernel : int
            A kernel size to smooth the centroid positions, and correct for long term trends. Must be odd. Default 21.
        aperture_mask : string
            An aperture mask type to perform the test on. See `lk.TargetPixelFile._parse_aperture_mask`. 'all' will use all pixels,
            `pipeline` will use the pipeline mask, and `threshold` will use all contiguous pixels above a 3 sigma threshold.
        plot : bool
            Whether or not to plot results.

    Returns
    -------
        r : dict
            Dictionary of results, containing 'figs' (a list of matplotlib figures) and 'pvalues', which is a list of tuples.
            Each tuple contains the p-value for the input planets. There are as many tuples as input TPFs.

    """
    if not isinstance(tpfs, list):
        tpfs = [tpfs]
    if not hasattr(periods, '__iter__'):
        periods = [periods]
    if not hasattr(t0s, '__iter__'):
        t0s = [t0s]
    if not hasattr(durs, '__iter__'):
        durs = [durs]

    nplanets = len(periods)
    r = {}
    r['figs'] = []
    r['pvalues'] = []
    for tpf in tpfs:
        crwd = tpfs[0].hdu[1].header['CROWDSAP']
        if crwd < 0.8:
            raise ValueError(f'Aperture is significantly crowded (CROWDSAP = {crwd}). This method will not work to centroid these cases.')

        aper = tpf._parse_aperture_mask(aperture_mask)
        mask = (np.abs((tpf.pos_corr1)) < 10) & ((np.gradient(tpf.pos_corr2)) < 10)
        mask &= np.nan_to_num(tpf.to_lightcurve(aperture_mask=aper).flux) != 0
        tpf = tpf[mask]
        lc = tpf.to_lightcurve(aperture_mask=aper)

        tmasks = []
        for period, t0, duration in zip(periods, t0s, durs):
            bls = lc.to_periodogram('bls', period=[period, period], duration=duration)
            t_mask = bls.get_transit_mask(period=period, transit_time=t0, duration=duration)
            tmasks.append(t_mask)
        tmasks= np.asarray(tmasks)
        Y, X = np.mgrid[:tpf.shape[1], :tpf.shape[2]]
        X = (X[aper][:, None] * np.ones(tpf.shape[0])).T
        Y = (Y[aper][:, None] * np.ones(tpf.shape[0])).T
        X = (X[:, :, None] * np.ones(40))
        Y = (Y[:, :, None] * np.ones(40))
        fe  = np.asarray([np.random.normal(tpf.flux[:, aper], tpf.flux_err[:, aper]) for idx in range(40)]).transpose([1, 2, 0])
        xcent = np.average(X, weights=fe, axis=1)
        ycent = np.average(Y, weights=fe, axis=1)

        xcent = np.asarray([np.nanmean(xcent, axis=1), np.nanstd(xcent, axis=1)]).T
        ycent = np.asarray([np.nanmean(ycent, axis=1), np.nanstd(ycent, axis=1)]).T



        breaks = np.where(np.diff(tpf.time) > 0.1)[0] + 1
        ms = [np.in1d(np.arange(len(tpf.time)), i) for i in np.array_split(np.arange(len(tpf.time)), breaks)]
        xtr, ytr = [], []
        for m in ms:
            xtr.append(convolve(xcent[:, 0][m], Gaussian1DKernel(kernel), boundary='extend'))
            ytr.append(convolve(ycent[:, 0][m], Gaussian1DKernel(kernel), boundary='extend'))

        xtr = np.hstack(xtr)
        ytr = np.hstack(ytr)

        xsamps = np.random.normal(xcent[:, 0] - xtr, xcent[:, 1], size=(50, len(xcent)))
        ysamps = np.random.normal(ycent[:, 0] - ytr, ycent[:, 1], size=(50, len(ycent)))

        if plot:
            with plt.style.context('seaborn-white'):
                fig, axs = plt.subplots(1, nplanets, figsize=(4*nplanets, 4), sharex=True, sharey=True, facecolor='w')
                if not hasattr(axs, '__iter__'):
                    axs = [axs]
        letter = 'bcd'
        pvalues = []
        for idx in range(nplanets):
            k1 = (tmasks).all(axis=0)
            k2 = ~tmasks[idx]
#            axs[idx].errorbar(xcent[:, 0][k1] - xtr[k1], ycent[:, 0][k1] - ytr[k1], xerr=xcent[:, 1][k1], yerr=ycent[:, 1][k1], c='k', ls='', lw=0.3, label='No Planet Cadences')
            if plot:
                with plt.style.context('seaborn-white'):
                    corner.hist2d(xcent[:, 0][k1] - xtr[k1], ycent[:, 0][k1] - ytr[k1], ax=axs[idx])
                    axs[idx].errorbar(xcent[:, 0][k2] - xtr[k2], ycent[:, 0][k2] - ytr[k2], xerr=xcent[:, 1][k2], yerr=ycent[:, 1][k2], c=f'C{idx}', lw=1, ls='', label=f'Transit {letter[idx]} Cadences')
                    axs[idx].set(xlabel='X Pixel Centroid', title=f'Transit {letter[idx]}')
                    axs[idx].legend(loc='upper left')
            ps = []
            for x1, y1 in zip(xsamps, ysamps):
                px = ttest_ind(x1[k1], x1[k2], equal_var = False)
                py = ttest_ind(y1[k1], y1[k2], equal_var = False)
                ps.append(np.mean([px.pvalue, py.pvalue]))
            pvalue = np.mean(ps)
            if plot:
                with plt.style.context('seaborn-white'):
                    if pvalue > 0.05:
                        axs[idx].text(0.975, 0.05, f'No Significant Offset (p-value: {pvalue:.2E})', horizontalalignment='right',
                                              verticalalignment='center', transform=axs[idx].transAxes)
                    else:
                        axs[idx].text(0.975, 0.05, f'Offset Detected(p-value: {pvalue:.2E})', horizontalalignment='right',
                                              verticalalignment='center', transform=axs[idx].transAxes)
                    plt.suptitle(_label(tpf))

                    axs[0].set_ylabel('Y Pixel Centroid')
                    plt.subplots_adjust(wspace=0)
                    r['figs'].append(fig)

            pvalues.append(pvalue)
        r['pvalues'].append(tuple(pvalues))
    return r
