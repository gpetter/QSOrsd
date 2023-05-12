from corrfunc_helper import twoPointCFs, plots
from astropy.table import Table
import numpy as np
import pickle
eboss_zs = [1.1, 1.4, 1.7, 2.]
boss_zs = [2.5, 3.0]

datadir = '/home/graysonpetter/ssd/Dartmouth/data/lss/'
plotdir = '/home/graysonpetter/Dropbox/rsdplots/'

lopercentile, hipercentile = 10, 90



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

def auto_all_eboss(rpscales, pimax, pibinsize, mbh_xcorr=False, lum_xcorr=False):
    qso = Table.read(datadir + 'eBOSS_QSO/eBOSS_QSO.fits')
    rand = Table.read(datadir + 'eBOSS_QSO/eBOSS_QSO_randoms.fits')

    cf = twoPointCFs.autocorr_cat(rpscales, qso, rand,
                                  nthreads=16, estimator='LS', pimax=pimax, dpi=pibinsize, nbootstrap=500)
    with open('results/cfs/ebossqso_cf.pickle', 'wb') as f:
        pickle.dump(cf, f)
    fig = cf['2dplot']
    fig.savefig(plotdir + 'all.pdf')

    if mbh_xcorr:
        highbh = rolling_percentile_selection(qso, 'MBH', minpercentile=hipercentile, nzbins=30)
        lobh = rolling_percentile_selection(qso, 'MBH', minpercentile=0, maxpercentile=lopercentile, nzbins=30)
        hicf = twoPointCFs.crosscorr_cats(rpscales, qso, highbh, rand,
                                          nthreads=16, estimator='Peebles',
                                          pimax=pimax, dpi=pibinsize, nbootstrap=500)
        locf = twoPointCFs.crosscorr_cats(rpscales, qso, lobh, rand,
                                          nthreads=16, estimator='Peebles',
                                          pimax=pimax, dpi=pibinsize, nbootstrap=500)
        with open('results/cfs/ebossqso_highbh.pickle', 'wb') as f:
            pickle.dump(hicf, f)
        with open('results/cfs/ebossqso_lobh.pickle', 'wb') as f:
            pickle.dump(locf, f)
        fig = plots.plotmultiple_2d_corr_func([cf['xi_rp_pi'], hicf['xi_rp_pi']])
        fig.savefig(plotdir + 'all_hibhmass.pdf')
        fig = plots.plotmultiple_2d_corr_func([cf['xi_rp_pi'], locf['xi_rp_pi']])
        fig.savefig(plotdir + 'all_lobhmass.pdf')
    if lum_xcorr:
        highlum = rolling_percentile_selection(qso, 'Lbol', minpercentile=hipercentile, nzbins=30)
        lolum = rolling_percentile_selection(qso, 'Lbol', minpercentile=0, maxpercentile=lopercentile, nzbins=30)
        hilumcf = twoPointCFs.crosscorr_cats(rpscales, qso, highlum, rand,
                                          nthreads=16, estimator='Peebles',
                                          pimax=pimax, dpi=pibinsize, nbootstrap=500)
        lolumcf = twoPointCFs.crosscorr_cats(rpscales, qso, lolum, rand,
                                          nthreads=16, estimator='Peebles',
                                          pimax=pimax, dpi=pibinsize, nbootstrap=500)
        with open('results/cfs/ebossqso_highbh.pickle', 'wb') as f:
            pickle.dump(hilumcf, f)
        with open('results/cfs/ebossqso_lobh.pickle', 'wb') as f:
            pickle.dump(lolumcf, f)
        fig = plots.plotmultiple_2d_corr_func([cf['xi_rp_pi'], hilumcf['xi_rp_pi']])
        fig.savefig(plotdir + 'all_hilum.pdf')
        fig = plots.plotmultiple_2d_corr_func([cf['xi_rp_pi'], lolumcf['xi_rp_pi']])
        fig.savefig(plotdir + 'all_lolum.pdf')

#auto_all_eboss(np.logspace(0., 1.5, 11), pimax=30, pibinsize=3, mbh_xcorr=True, lum_xcorr=True)

def auto_all_boss(rpscales, pimax, pibinsize):
    qso = Table.read(datadir + 'BOSS_QSO/BOSS_QSO.fits')
    rand = Table.read(datadir + 'BOSS_QSO/BOSS_QSO_randoms.fits')

    cf = twoPointCFs.autocorr_cat(rpscales, qso, rand,
                                  nthreads=16, estimator='LS', pimax=pimax, dpi=pibinsize, nbootstrap=500)



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


