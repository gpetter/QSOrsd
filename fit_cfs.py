import abacus_mcmc
import numpy as np
import yaml
import pickle
from abacusnbody.hod.abacus_hod import AbacusHOD

import manager
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

small = False
if small:
    sim_params['sim_name'] = 'AbacusSummit_small_c000_ph3000'
else:
    sim_params['sim_name'] = 'AbacusSummit_base_c000_ph000'

#eboss_zs = [1.1, 1.4, 1.7, 2.]
eboss_zs = [1.7, 2.]
boss_zs = [2.5, 3.1]


n_chunks = 11




def fit_autocorrs_z_centonly(nwalkers, niter, do_eboss=True, do_boss=False, assembly_bias=False):
    #sim_params['subsample_dir'] = '/home/graysonpetter/extssd/subsample_noparticles/'
    eboss = Table.read('/home/graysonpetter/ssd/Dartmouth/data/lss/eBOSS_QSO/eBOSS_QSO.fits')

    if do_eboss:
        for z in eboss_zs:
            with open('results/cfs/z%s_cf.pickle' % z, 'rb') as f:
                cf2d = pickle.load(f)
            qsoz = eboss[np.where((eboss['Z'] < (z+0.15)) & (eboss['Z'] > (z - 0.15)))]
            ball = manager.load_sim(z=z, cent_only=True, sat_only=False, nslabs=3, small=False, z_errs=qsoz['sigZ'])
            #sim_params['z_mock'] = z
            #ball = AbacusHOD(sim_params, HOD_params, clustering_params, chunk=0, n_chunks=n_chunks, delta_z_dist=qsoz['sigZ'])
            param_names = ['logM_cut', 'alpha_c']
            init_params = [12.25, .8]
            gauss_priors = np.array([[12.25, 0.15], [.5, 0.3]])

            fit_no_ab = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                        cf2d=cf2d,
                                                        param_names=param_names, ball=ball, initial_params=init_params,
                                                        pool=None, nthread=16, gausspriors=gauss_priors)
            flatchain, centers, lobounds, hibounds = fit_no_ab
            cornerplots.cornerplot(flatchain, [r'log $M_{\rm min}$', r'$\alpha_{c}$'], 'z%s_2param' % z)
            outdict = {'params': param_names, 'chain': flatchain, 'values': centers,
                       'sig_lo': lobounds, 'sig_up': hibounds}
            with open('results/fits/z%s_centonly.pickle' % z, 'wb') as f:
                pickle.dump(outdict, f)


            if assembly_bias:
                ball = manager.reset_params(ball)
                #chainab, valsab, loerrab, hierrab = fit_no_ab
                param_names += ['Bcent']
                init_params = [centers[0], centers[1], 0.]
                fit_w_ab = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                          cf2d=cf2d,
                                                          param_names=param_names, ball=ball, initial_params=init_params,
                                                          pool=None, nthread=16)
                flatchain, centers, lobounds, hibounds = fit_w_ab
                cornerplots.cornerplot(flatchain, [r'log $M_{\rm min}$', r'$\alpha_{c}$', r'$B_{cent}$'], 'z%s_3param' % z)
                outdict = {'params': param_names, 'chain': flatchain, 'values': centers,
                           'sig_lo': lobounds, 'sig_up': hibounds}
                with open('results/fits/z%s_centonly_AB.pickle' % z, 'wb') as f:
                    pickle.dump(outdict, f)
    if do_boss:
        boss = Table.read('/home/graysonpetter/ssd/Dartmouth/data/lss/BOSS_QSO/BOSS_QSO.fits')
        for z in boss_zs:
            with open('results/cfs/z%s_cf.pickle' % z, 'rb') as f:
                cf2d = pickle.load(f)
            qsoz = boss[np.where((boss['Z'] < (z+0.15)) & (boss['Z'] > (z - 0.15)))]
            sim_params['z_mock'] = z
            ball = AbacusHOD(sim_params, HOD_params, clustering_params, chunk=0, n_chunks=n_chunks, delta_z_dist=qsoz['sigZ'])
            param_names = ['logM_cut', 'alpha_c']
            init_params = [12.2, 1.]


            fit_no_ab = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                        cf2d=cf2d,
                                                        param_names=param_names, ball=ball, initial_params=init_params,
                                                        pool=None, nthread=16)
            flatchain, centers, lobounds, hibounds = fit_no_ab
            cornerplots.cornerplot(fit_no_ab[0], [r'log $M_{\rm min}$', r'$\alpha_{c}$'], 'z%s_2param' % z)
            outdict = {'params': param_names, 'chain': flatchain, 'values': centers,
                       'sig_lo': lobounds, 'sig_up': hibounds}
            with open('results/fits/z%s_centonly.pickle' % z, 'wb') as f:
                pickle.dump(outdict, f)

            #chainab, valsab, loerrab, hierrab = fit_no_ab
            """param_names += 'Bcent'
            init_params = [12.2, 1., 0.]
            fit_w_ab = abacus_mcmc.sample_abacus_hod(nwalkers=nwalkers, niter=niter,
                                                      cf2d=cf2d,
                                                      param_names=param_names, ball=ball, initial_params=init_params,
                                                      pool=None, nthread=16)"""

fit_autocorrs_z_centonly(10, 1000, do_boss=True, do_eboss=False)

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
