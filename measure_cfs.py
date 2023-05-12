from corrfunc_helper import twoPointCFs, plots
from astropy.table import Table
import numpy as np
import manager


datadir = '/home/graysonpetter/ssd/Dartmouth/data/lss/'
plotdir = '/home/graysonpetter/Dropbox/rsdplots/'

lopercentile, hipercentile = 10, 90

def all_clustering(rpscales, cat, randcat, pimax, dpi, s_scales, mubins, wedges, cat2=None):
    if cat2 is None:
        cf = twoPointCFs.autocorr_cat(rpscales, cat, randcat, nthreads=16, estimator='LS',
                                      pimax=pimax, dpi=dpi, nbootstrap=500)
        cf_s_mu = twoPointCFs.autocorr_cat(s_scales, cat, randcat, nthreads=16, estimator='LS', mubins=mubins,
                                           nbootstrap=500, wedges=wedges)
    else:
        cf = twoPointCFs.crosscorr_cats(rpscales, cat, cat2, randcat, nthreads=16, estimator='LS',
                                      pimax=pimax, dpi=dpi, nbootstrap=500)
        cf_s_mu = twoPointCFs.crosscorr_cats(s_scales, cat, cat2, randcat, nthreads=16, estimator='LS', mubins=mubins,
                                           nbootstrap=500, wedges=wedges)

    cf.pop('plot')
    keylist = ['s', 's_bins', 'mono', 'quad', 'mono_err', 'quad_err']
    for thiskey in keylist:
        cf[thiskey] = cf_s_mu[thiskey]
    fig = cf.pop('2dplot')
    return cf, fig

def rolling_percentile_selection(cat, prop, minpercentile, maxpercentile=100, nzbins=100):
    """
    choose highest nth percentile of e.g. luminosity in bins of redshift
    """
    minz, maxz = np.min(cat['Z']), np.max(cat['Z'])
    zbins = np.linspace(minz, maxz, nzbins)
    idxs = []
    for j in range(len(zbins)-1):
        catinbin = cat[np.where((cat['Z'] > zbins[j]) & (cat['Z'] <= zbins[j+1]))]
        thresh = np.percentile(catinbin[prop], minpercentile)
        hithresh = np.percentile(catinbin[prop], maxpercentile)
        idxs += list(np.where((cat[prop] > thresh) & (cat[prop] <= hithresh) &
                              (cat['Z'] > zbins[j]) & (cat['Z'] <= zbins[j+1]))[0])

    newcat = cat[np.array(idxs)]
    return newcat

def measure_all_cfs(rpmax, nrp, pimax, pibinsize, s_scales, mubins, wedges, eboss=True, mbh_xcorr=False, lum_xcorr=False):
    if eboss:
        qso = Table.read(datadir + 'eBOSS_QSO/eBOSS_QSO.fits')
        rand = Table.read(datadir + 'eBOSS_QSO/eBOSS_QSO_randoms.fits')
        dz = 0.15
        zlist = [1.1, 1.4, 1.7, 2.]
    else:
        qso = Table.read(datadir + 'BOSS_QSO/BOSS_QSO.fits')
        rand = Table.read(datadir + 'BOSS_QSO/BOSS_QSO_randoms.fits')
        dz = 0.3
        zlist = [2.5, 3.1]

    for z in zlist:
        qsoz = qso[np.where((qso['Z'] > (z - dz)) * (qso['Z'] <= (z + dz)))]
        randz = rand[np.where((rand['Z'] > (z - dz)) * (rand['Z'] <= (z + dz)))]
        # avoid fiber collision limit
        minrp = manager.min_scale(np.max(qsoz['Z']))
        rpscales = np.logspace(np.log10(minrp), np.log10(rpmax), nrp+1)

        cf, fig = all_clustering(rpscales=rpscales, cat=qsoz, randcat=randz, pimax=pimax, dpi=pibinsize,
                                 s_scales=s_scales, mubins=mubins, wedges=wedges)
        fig.savefig(plotdir + 'z%s.pdf' % z)
        manager.write_pickle('results/cfs/%s_cf' % z, cf)

        if mbh_xcorr:
            highbh = rolling_percentile_selection(qsoz, 'MBH', minpercentile=hipercentile, nzbins=30)
            lobh = rolling_percentile_selection(qsoz, 'MBH', minpercentile=0, maxpercentile=lopercentile, nzbins=30)
            hicf = twoPointCFs.crosscorr_cats(rpscales, qsoz, highbh, randz,
                                              nthreads=16, estimator='Peebles',
                                              pimax=pimax, dpi=pibinsize, nbootstrap=500)
            locf = twoPointCFs.crosscorr_cats(rpscales, qsoz, lobh, randz,
                                              nthreads=16, estimator='Peebles',
                                              pimax=pimax, dpi=pibinsize, nbootstrap=500)
            fig = plots.plotmultiple_2d_corr_func([locf['xi_rp_pi'], hicf['xi_rp_pi']])
            fig.savefig(plotdir + 'z%s_bh.pdf' % z)
            locf.pop('2dplot'), locf.pop('plot'), hicf.pop('2dplot'), hicf.pop('plot')

            manager.write_pickle('results/cfs/%s_lobh_cf' % z, locf)
            manager.write_pickle('results/cfs/%s_hibh_cf' % z, hicf)
        if lum_xcorr:
            hilum = rolling_percentile_selection(qsoz, 'Lbol', minpercentile=hipercentile, nzbins=30)
            lolum = rolling_percentile_selection(qsoz, 'Lbol', minpercentile=0, maxpercentile=lopercentile, nzbins=30)
            hicf = twoPointCFs.crosscorr_cats(rpscales, qsoz, hilum, randz,
                                              nthreads=16, estimator='Peebles',
                                              pimax=pimax, dpi=pibinsize, nbootstrap=500)
            locf = twoPointCFs.crosscorr_cats(rpscales, qsoz, lolum, randz,
                                              nthreads=16, estimator='Peebles',
                                              pimax=pimax, dpi=pibinsize, nbootstrap=500)
            fig = plots.plotmultiple_2d_corr_func([locf['xi_rp_pi'], hicf['xi_rp_pi']])
            fig.savefig(plotdir + 'z%s_lum.pdf' % z)
            locf.pop('2dplot'), locf.pop('plot'), hicf.pop('2dplot'), hicf.pop('plot')

            manager.write_pickle('results/cfs/%s_lolum_cf' % z, locf)
            manager.write_pickle('results/cfs/%s_hilum_cf' % z, hicf)

measure_all_cfs(rpmax=25., nrp=5, pimax=25, pibinsize=5, s_scales=np.logspace(0., 1.4, 20), mubins=20, wedges=5,
                eboss=True, mbh_xcorr=False, lum_xcorr=False)




def qso_x_lrgs(rpscales, pimax, pibinsize):
    qso = Table.read(datadir + 'eBOSS_QSO/eBOSS_QSO.fits')
    qsorand = Table.read(datadir + 'eBOSS_QSO/eBOSS_QSO_randoms.fits')
    lrg = Table.read(datadir + 'eBOSS_LRG/eBOSS_LRG.fits')
    lrgrand = Table.read(datadir + 'eBOSS_LRG/eBOSS_LRG_randoms.fits')

    # get only qsos, lrgs in overlap at z=0.8 to z=1.
    qso = qso[np.where(qso['Z'] < 1.)]
    qsorand = qsorand[np.where(qsorand['Z'] < 1.)]

    lrg = lrg[np.where(lrg['Z'] > .8)]
    lrgrand = lrgrand[np.where(lrgrand['Z'] > .8)]

    lrgcf = twoPointCFs.autocorr_cat(rpscales, lrg, lrgrand,
                                     nthreads=16, estimator='LS', pimax=pimax, dpi=pibinsize, nbootstrap=500)

    qsolrgcf = twoPointCFs.crosscorr_cats(rpscales, lrg, qso, lrgrand,
                                          nthreads=16, estimator='Peebles', pimax=pimax, dpi=pibinsize, nbootstrap=500)


