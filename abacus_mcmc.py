import numpy as np
import emcee
import mcmc

param_bounds = {'logM_cut': (11.7, 12.6), 'logM1': (13.3, 14.5), 'alpha_c': (0., 1.5), 's': (-1., 1.),
                    's_v': (-1., 1.), 'Bcent': (-1., 1.), 'Bsat': (-1., 1.)}

default_param_priors = {'logM_cut': (12.2, 0.5), 'alpha_c': (0.5, 0.3), 's': (0., 0.3),
                    's_v': (0., 0.3), 'Bcent': (0., 0.3), 'Bsat': (0., 0.3)}

def ln_prior(theta, param_names, gausspriors):
    priorsum = 0
    for j in range(len(theta)):
        paramval = theta[j]
        if (paramval <= param_bounds[param_names[j]][0]) or (paramval >= param_bounds[param_names[j]][1]):
            priorsum -= np.inf
            return priorsum
        if gausspriors is not None:
            mu_j, sig_j = gausspriors[j]
            priorsum -= 0.5 * ((paramval - mu_j) / sig_j) ** 2
    return priorsum


def ln_prob_abacushod(theta, param_names, cf2d_observed, ball, force_qso_satellites, gausspriors, nthread=24):
    rp_bins = cf2d_observed['rp_bins']
    pimax = int(cf2d_observed['pimax'])
    pi_bin_size = int(cf2d_observed['dpi'])
    cf_obs = np.array(cf2d_observed['xi_rp_pi']).ravel()
    cf_obs_err = np.array(cf2d_observed['xi_rp_pi_err']).ravel()

    prior = ln_prior(theta, param_names, gausspriors)
    if not np.isfinite(prior):
        return -np.inf
    else:
        for j in range(len(theta)):
            ball.tracers['QSO'][param_names[j]] = theta[j]
        mock_dict = ball.run_hod(ball.tracers, want_rsd=True, write_to_disk=False, Nthread=nthread, verbose=False,
                                 force_qso_satellites=force_qso_satellites)
        xirppi = ball.compute_xirppi(mock_dict=mock_dict, rpbins=rp_bins,
                                     pimax=pimax, pi_bin_size=pi_bin_size)['QSO_QSO'].ravel()
        residual = cf_obs - xirppi
        likely = mcmc.ln_likelihood(residual=residual, yerr=cf_obs_err)
        return likely + prior

def ln_prob_abacushod_cross(theta, param_names, cf_auto, cf_cross, ball, force_qso_satellites, gausspriors_auto, gausspriors_cross, nthread=24):
    rp_bins = cf_auto['rp_bins']
    pimax = int(cf_auto['pimax'])
    pi_bin_size = int(cf_auto['dpi'])
    cf_obs = np.concatenate([np.array(cf_auto['xi_rp_pi']).ravel(), np.array(cf_cross['xi_rp_pi']).ravel()])
    cf_obs_err = np.concatenate([np.array(cf_auto['xi_rp_pi_err']).ravel(), np.array(cf_cross['xi_rp_pi_err']).ravel()])
    nparams = len(param_names)


    prior = ln_prior(theta[], param_names, gausspriors_auto) + ln_prior(theta[], param_names, gausspriors_cross)
    if not np.isfinite(prior):
        return -np.inf
    else:
        for j in range(len(param_names)):
            ball.tracers['LRG'][param_names[j]] = theta[j+nparams]
            ball.tracers['QSO'][param_names[j]] = theta[j]
        mock_dict = ball.run_hod(ball.tracers, want_rsd=True, write_to_disk=False, Nthread=nthread, verbose=False,
                                 force_qso_satellites=force_qso_satellites)
        xirppi = ball.compute_xirppi(mock_dict=mock_dict, rpbins=rp_bins,
                                     pimax=pimax, pi_bin_size=pi_bin_size)['LRG_QSO'].ravel()
        residual = cf_obs - xirppi
        likely = mcmc.ln_likelihood(residual=residual, yerr=cf_obs_err)
        return likely + prior


def sample_abacus_hod(nwalkers, niter, cf2d, param_names, ball, initial_params=None, pool=None,
                      force_qso_satellites=False, gausspriors=None, nthread=1):

    if gausspriors is None:
        gausspriors = [[default_param_priors[name][0], default_param_priors[name][1]] for name in param_names]
        init_scatter = 2e-1
    else:
        init_scatter = np.array(gausspriors)[:, 1] / 2.

    ndim = len(param_names)
    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    ln_prob_abacushod,
                                    args=[param_names, cf2d, ball, force_qso_satellites, gausspriors, nthread],
                                    pool=pool)

    # start walkers near least squares fit position with random gaussian offsets
    pos = np.array(initial_params) + np.random.normal(size=(sampler.nwalkers, sampler.ndim)) * init_scatter
    sampler.run_mcmc(pos, niter, progress=True)


    flatchain = sampler.get_chain(discard=int(niter/10), flat=True)
    #blobs = sampler.get_blobs(discard=10, flat=True)

    centervals, lowerrs, higherrs = [], [], []
    for i in range(ndim):
        post = np.percentile(flatchain[:, i], [16, 50, 84])
        q = np.diff(post)
        centervals.append(post[1])
        lowerrs.append(q[0])
        higherrs.append(q[1])

    return flatchain, centervals, lowerrs, higherrs

def sample_abacus_hod_xcorr(nwalkers, niter, cfauto, cfcross, param_names, ball, initial_params=None, pool=None,
                      force_qso_satellites=False, gausspriors_auto=None, gausspriors_cross=None, nthread=1):

    if gausspriors_auto is None:
        gausspriors_auto = [[default_param_priors[name][0], default_param_priors[name][1]] for name in param_names]
        init_scatter_auto = 2e-1
    else:
        init_scatter_auto = np.array(gausspriors_auto)[:, 1] / 2.
    if gausspriors_cross is None:
        gausspriors_cross = [[default_param_priors[name][0], default_param_priors[name][1]] for name in param_names]
        init_scatter_cross = 2e-1
    else:
        init_scatter_cross = np.array(gausspriors_cross)[:, 1] / 2.

    ndim = len(param_names)
    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    ln_prob_abacushod,
                                    args=[param_names, cf2d, ball, force_qso_satellites, gausspriors, nthread],
                                    pool=pool)

    # start walkers near least squares fit position with random gaussian offsets
    pos = np.array(initial_params) + np.random.normal(size=(sampler.nwalkers, sampler.ndim)) * init_scatter
    sampler.run_mcmc(pos, niter, progress=True)


    flatchain = sampler.get_chain(discard=int(niter/10), flat=True)
    #blobs = sampler.get_blobs(discard=10, flat=True)

    centervals, lowerrs, higherrs = [], [], []
    for i in range(ndim):
        post = np.percentile(flatchain[:, i], [16, 50, 84])
        q = np.diff(post)
        centervals.append(post[1])
        lowerrs.append(q[0])
        higherrs.append(q[1])

    return flatchain, centervals, lowerrs, higherrs