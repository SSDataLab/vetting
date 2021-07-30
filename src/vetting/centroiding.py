import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.units as u
import lightkurve as lk

from scipy.stats import ttest_ind
import corner


def _label(tpf):
    if hasattr(tpf, "quarter"):
        return f"{tpf.to_lightcurve().label}, Quarter {tpf.quarter}"
    elif hasattr(tpf, "campaign"):
        return f"{tpf.to_lightcurve().label}, Campaign {tpf.campaign}"
    elif hasattr(tpf, "sector"):
        return f"{tpf.to_lightcurve().label}, Sector {tpf.sector}"
    else:
        return "{tpf.to_lightcurve().label}"


def weighted_average(values, weights, axis=None):
    mean = np.average(values, weights=weights, axis=axis)
    error = np.average((values - mean) ** 2, weights=weights, axis=axis) ** 0.5
    return mean, error / len(values) ** 0.5


def centroid_test(
    tpfs,
    periods,
    t0s,
    durs,
    kernel=21,
    aperture_mask="pipeline",
    plot=True,
    nsamp=100,
    transit_depths=None,
    labels=None,
):
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
        nsamp : int
            Number of samples to make when testing (default 50)
        transit_depths: None, List of floats
            Optional. If given, will calculate the separation at which contaminants can be ruled out. Must be between 0 and 1.
        labels: None, list of strings, string
            Optional. Label for each planet in plots. If None, will generate labels in sequential order. e.g. pass `'bcd'` to label
            planets. Must same length as periods.
    Returns
    -------
        r : dict
            Dictionary of results, containing 'figs' (a list of matplotlib figures) and 'pvalues', which is a list of tuples.
            Each tuple contains the p-value for the input planets. There are as many tuples as input TPFs.

    """
    if isinstance(tpfs, lk.targetpixelfile.TargetPixelFile):
        tpfs = [tpfs]
    if isinstance(tpfs, lk.collections.TargetPixelFileCollection):
        tpfs = [tpf for tpf in tpfs]
    if not hasattr(periods, "__iter__"):
        periods = [periods]
    if not hasattr(t0s, "__iter__"):
        t0s = [t0s]
    if not hasattr(durs, "__iter__"):
        durs = [durs]
    if transit_depths is not None:
        if not hasattr(transit_depths, "__iter__"):
            transit_depths = [transit_depths]
        if not len(transit_depths) == len(periods):
            raise ValueError("Please pass same length `transit_depth` as `periods`")
        for transit_depth in transit_depths:
            if (transit_depth < 0) | (transit_depth > 1):
                raise ValueError("`transit_depth` must be > 0 and < 1. ")

    if labels is not None:
        if not hasattr(labels, "__iter__"):
            labels = [labels]
        if not len(labels) == len(periods):
            raise ValueError("Please pass same length `labels` as `periods`")
    else:
        labels = "bcdefghijklmnopqrstu"

    nplanets = len(periods)
    r = {}
    for key in ["figs", "pvalues", "centroid_offset_detected"]:
        r[key] = []
    if transit_depths is not None:
        for key in ["1sigma_error"]:
            r[key] = []

    offsets = np.logspace(-6, np.log10(3), 100)[::-1]
    for tpf in tpfs:
        if tpf.mission.lower() in ["kepler", "ktwo", "k2"]:
            pixel_scale = 4

        elif tpf.mission.lower() == "tess":
            pixel_scale = 27
        else:
            raise ValueError(
                "Can not understand mission keyword in TPF to assign pixel scale."
            )
        crwd = tpf.hdu[1].header["CROWDSAP"]
        if crwd is not None:
            if crwd < 0.8:
                raise ValueError(
                    f"Aperture is significantly crowded (CROWDSAP = {crwd}). This method will not work to centroid these cases."
                )

        aper = tpf._parse_aperture_mask(aperture_mask)
        mask = (np.abs((tpf.pos_corr1)) < 10) & ((np.gradient(tpf.pos_corr2)) < 10)
        mask &= np.nan_to_num(tpf.to_lightcurve(aperture_mask=aper).flux) != 0
        tpf = tpf[mask]
        lc = tpf.to_lightcurve(aperture_mask=aper)

        tmasks = []
        for period, t0, duration in zip(periods, t0s, durs):
            bls = lc.to_periodogram("bls", period=[period, period], duration=duration)
            t_mask = ~bls.get_transit_mask(
                period=period, transit_time=t0, duration=duration
            )
            tmasks.append(t_mask)
        tmasks = np.asarray(tmasks)
        Y, X = np.mgrid[: tpf.shape[1], : tpf.shape[2]]
        X = (X[aper][:, None] * np.ones(tpf.shape[0])).T
        Y = (Y[aper][:, None] * np.ones(tpf.shape[0])).T

        X = X[:, :, None] * np.ones(nsamp)
        Y = Y[:, :, None] * np.ones(nsamp)
        fe = np.asarray(
            [
                np.random.normal(tpf.flux[:, aper], tpf.flux_err[:, aper])
                for idx in range(nsamp)
            ]
        ).transpose([1, 2, 0])

        xcent = np.average(X, weights=fe, axis=1)
        ycent = np.average(Y, weights=fe, axis=1)
        xcent = np.asarray([np.nanmean(xcent, axis=1), np.nanstd(xcent, axis=1)]).T
        ycent = np.asarray([np.nanmean(ycent, axis=1), np.nanstd(ycent, axis=1)]).T
        import pdb

        # If the mission is K2, we need to use SFF to detrend the centroids.
        if tpf.mission.lower() in ["ktwo", "k2"]:
            xlc = lk.KeplerLightCurve(
                time=tpf.time,
                flux=xcent[:, 0],
                flux_err=xcent[:, 1],
                centroid_col=tpf.pos_corr1,
                centroid_row=tpf.pos_corr2,
                targetid="x",
            )
            s = lk.SFFCorrector(xlc)
            s.correct(
                windows=20,
                bins=10,
                cadence_mask=t_mask,
                propagate_errors=True,
                timescale=10,
            )
            xcent[:, 0] -= s.model_lc.flux.value

            ylc = lk.KeplerLightCurve(
                time=tpf.time,
                flux=ycent[:, 0],
                flux_err=ycent[:, 1],
                centroid_col=tpf.pos_corr1,
                centroid_row=tpf.pos_corr2,
                targetid="y",
            )
            s = lk.SFFCorrector(ylc)
            s.correct(
                windows=20,
                bins=10,
                cadence_mask=t_mask,
                propagate_errors=True,
                timescale=10,
            )
            ycent[:, 0] -= s.model_lc.flux.value

        dt = np.diff(tpf.time.jd)
        breaks = np.where(dt > 10 * np.median(dt))[0] + 1
        ms = [
            np.in1d(np.arange(len(tpf.time)), i)
            for i in np.array_split(np.arange(len(tpf.time)), breaks)
        ]
        xtr, ytr = [], []
        for m in ms:
            xtr.append(
                convolve(xcent[:, 0][m], Gaussian1DKernel(kernel), boundary="extend")
            )
            ytr.append(
                convolve(ycent[:, 0][m], Gaussian1DKernel(kernel), boundary="extend")
            )

        xtr = np.hstack(xtr)
        ytr = np.hstack(ytr)

        xsamps = np.random.normal(
            xcent[:, 0] - xtr, xcent[:, 1], size=(nsamp, len(xcent))
        )
        ysamps = np.random.normal(
            ycent[:, 0] - ytr, ycent[:, 1], size=(nsamp, len(ycent))
        )

        if plot:
            with plt.style.context("seaborn-white"):
                fig, axs = plt.subplots(
                    1,
                    nplanets,
                    figsize=(4 * nplanets, 4),
                    sharex=True,
                    sharey=True,
                    facecolor="w",
                )
                if not hasattr(axs, "__iter__"):
                    axs = [axs]

        pvalues, sigma1, centroid_offset_detected = [], [], []
        for idx in range(nplanets):
            # NO Transits
            k1 = (tmasks).all(axis=0)
            # Transits of planet IDX
            k2 = ~tmasks[idx]
            if plot:
                if transit_depths is not None:
                    scale = pixel_scale / transit_depths[idx]
                    axs[idx].set(
                        xlabel='X Centroid ["]', title=f"Transit {labels[idx]}"
                    )
                else:
                    scale = 1
                    axs[idx].set(
                        xlabel="X Centroid [pixel]", title=f"Transit {labels[idx]}"
                    )

                with plt.style.context("seaborn-white"):
                    xc1 = (xcent[:, 0][k1] - xtr[k1]) * scale
                    yc1 = (ycent[:, 0][k1] - ytr[k1]) * scale
                    corner.hist2d(
                        xc1,
                        yc1,
                        ax=axs[idx],
                        range=[
                            np.nanpercentile(xc1, (1, 99)),
                            np.nanpercentile(yc1, (1, 99)),
                        ],
                    )
                    axs[idx].errorbar(
                        (xcent[:, 0][k2] - xtr[k2]) * scale,
                        (ycent[:, 0][k2] - ytr[k2]) * scale,
                        xerr=(xcent[:, 1][k2]) * scale,
                        yerr=(ycent[:, 1][k2]) * scale,
                        marker=".",
                        markersize=1,
                        c=f"C{idx}",
                        lw=1,
                        ls="",
                        label=f"Transit {labels[idx]} Cadences",
                    )
                    axs[idx].legend(loc="upper left")

            ps = []
            # ps1 = []
            for x1, y1 in zip(xsamps, ysamps):
                px = ttest_ind(x1[k1], x1[k2], equal_var=False)
                py = ttest_ind(y1[k1], y1[k2], equal_var=False)
                ps.append(np.mean([px.pvalue, py.pvalue]))

            if transit_depths is not None:
                # Weighted average and weighted standard deviation of out of transit
                a1 = weighted_average(xcent[k1, 0] - xtr[k1], 1 / xcent[k1, 1])
                b1 = weighted_average(ycent[k1, 0] - ytr[k1], 1 / ycent[k1, 1])
                # Weighted average and weighted standard deviation of out of transit
                a2 = weighted_average(xcent[k2, 0] - xtr[k2], 1 / xcent[k2, 1])
                b2 = weighted_average(ycent[k2, 0] - ytr[k2], 1 / ycent[k2, 1])
                pos_err = np.hypot(np.hypot(a1[1], b1[1]), np.hypot(a2[1], b2[1]))

            if transit_depths is not None:
                sigma1.append(pixel_scale * pos_err / transit_depths[idx])

            if k2.sum() == 0:
                pvalue = 1
            else:
                pvalue = np.mean(ps)
            pvalues.append(pvalue)
            if pvalue < 0.317:
                centroid_offset_detected.append(True)
            else:
                centroid_offset_detected.append(False)

            if plot:
                with plt.style.context("seaborn-white"):
                    if pvalue > 0.317:
                        label = f"No Significant Offset (p-value: {pvalue:.2E})"
                        if transit_depths is not None:
                            label = (
                                label
                                + f"\n1 Sigma Error: {(np.round(sigma1[idx], 2) * u.arcsecond).to_string(format='latex')}"
                            )
                        axs[idx].text(
                            0.975,
                            0.05,
                            label,
                            horizontalalignment="right",
                            verticalalignment="center",
                            transform=axs[idx].transAxes,
                        )
                    else:
                        axs[idx].text(
                            0.975,
                            0.05,
                            f"Offset Detected(p-value: {pvalue:.2E})",
                            horizontalalignment="right",
                            verticalalignment="center",
                            transform=axs[idx].transAxes,
                            color="r",
                        )
                    plt.suptitle(_label(tpf))
                    if transit_depths is not None:
                        axs[0].set_ylabel('Y Centroid ["]')
                    else:
                        axs[0].set_ylabel("Y Centroid [pixel]")
                    plt.subplots_adjust(wspace=0)

        if transit_depths is not None:
            r["1sigma_error"].append(np.asarray(sigma1) * u.arcsecond)
        r["centroid_offset_detected"].append(np.asarray(centroid_offset_detected))
        if plot:
            r["figs"].append(fig)
        r["pvalues"].append(tuple(pvalues))

    return r
