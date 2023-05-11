import abacus_mcmc
import numpy as np
import yaml
import pickle
from abacusnbody.hod.abacus_hod import AbacusHOD
from plotscripts import cornerplots
from astropy.table import Table

config = yaml.safe_load(open('abacus_hod.yaml'))
sim_params = config['sim_params']
HOD_params = config['HOD_params']
clustering_params = config['clustering_params']
want_rsd = HOD_params['want_rsd']
write_to_disk = HOD_params['write_to_disk']
HOD_params['QSO_params']['logM1'] = 16.
HOD_params['QSO_params']['sigma'] = 1e-4

eboss_zs = [1.1, 1.4, 1.7, 2.]

def decide_nchunks(z):
    from colossus.cosmology import cosmology
    from colossus.lss import mass_function
    mmin = 12.2
    masses = np.logspace(mmin, 16, 100)
    cosmology.setCosmology('planck18')
    mfunc_so = mass_function.massFunction(masses, z, mdef='200c', model='tinker08', q_out='dndlnM')
    # number of halos more massive than Mmin in the simulation box
    nhalos = (2000 ** 3) * np.trapz(mfunc_so, x=np.log(masses))
    # only need to keep enough slabs to contain ~10x the number of quasars at redshift for
    # model Poisson noise to be subdominant
    # quick calculation shows 3 slabs is more than enough at all redshifts, set n_chunks=11
n_chunks = 11

def fit_autocorrs_z_centonly(nwalkers, niter):
    sim_params['subsample_dir'] = '/home/graysonpetter/extssd/subsample_noparticles/'
    eboss = Table.read('/home/graysonpetter/ssd/Dartmouth/data/lss/eBOSS_QSO/eBOSS_QSO.fits')

    for z in eboss_zs:
        with open('results/cfs/z%s_cf.pickle' % z, 'rb') as f:
            cf2d = pickle.load(f)
        qsoz = eboss[np.where((eboss['Z'] < (z+0.15)) & (eboss['Z'] > (z - 0.15)))]
        sim_params['z_mock'] = z
        ball = AbacusHOD(sim_params, HOD_params, clustering_params, chunk=0, n_chunks=n_chunks, delta_z_dist=qsoz['sigZ'])
        param_names = ['logM_cut', 'alpha_c']
        init_params = [12.2, 1.]


        fit_no_ab = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                    cf2d=cf2d,
                                                    param_names=param_names, ball=ball, initial_params=init_params,
                                                    pool=None, nthread=16)
        cornerplots.cornerplot(fit_no_ab[0], [r'log $M_{\rm min}$', r'$\alpha_{c}$'], 'z%s_2param' % z)

        #chainab, valsab, loerrab, hierrab = fit_no_ab
        """param_names += 'Bcent'
        init_params = [12.2, 1., 0.]
        fit_w_ab = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                  cf2d=cf2d,
                                                  param_names=param_names, ball=ball, initial_params=init_params,
                                                  pool=None, nthread=16)"""

fit_autocorrs_z_centonly(10, 200)

def fit_autocorrs_z_satsonly(nwalkers, niter):
    sim_params['subsample_dir'] = '/home/graysonpetter/extssd/subsample_sats/'

    for z in eboss_zs:
        sim_params['z_mock'] = z
        ball = AbacusHOD(sim_params, HOD_params, clustering_params, chunk=0, n_chunks=n_chunks)
        param_names = ['logM_cut', 's_v']
        init_params = [12.2, 1.5]
        fit_velb = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                  cf2d=cf2d,
                                                  param_names=param_names, ball=ball, initial_params=init_params,
                                                  pool=None, nthread=1)
        # reset velocity bias to default of no bias
        ball.tracers['QSO']['s_v'] = 0.
        param_names = ['logM_cut', 's']
        init_params = [12.2, 0.5]
        fit_satprof = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                 cf2d=cf2d,
                                                 param_names=param_names, ball=ball, initial_params=init_params,
                                                 pool=None, nthread=1)

        # reset profile to default
        ball.tracers['QSO']['s'] = 0.
        param_names = ['logM_cut', 's_v', 'Bsat']
        init_params = [12.2, 1.5, 0.]
        fit_velb_ab = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                  cf2d=cf2d,
                                                  param_names=param_names, ball=ball, initial_params=init_params,
                                                  pool=None, nthread=1)

        # reset profile to default
        ball.tracers['QSO']['s_v'] = 0.
        param_names = ['logM_cut', 's', 'Bsat']
        init_params = [12.2, 1.5, 0.]
        fit_satprof_ab = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                  cf2d=cf2d,
                                                  param_names=param_names, ball=ball, initial_params=init_params,
                                                  pool=None, nthread=1)
