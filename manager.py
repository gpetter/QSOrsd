from math import floor
import numpy as np
import yaml
import pickle
from abacusnbody.hod.abacus_hod import AbacusHOD
from astropy.table import Table

config = yaml.safe_load(open('abacus_hod.yaml'))
sim_params = config['sim_params']
HOD_params = config['HOD_params']
clustering_params = config['clustering_params']
want_rsd = HOD_params['want_rsd']
write_to_disk = HOD_params['write_to_disk']

centonlyname = '/home/graysonpetter/extssd/subsample_noparticles/'
satsname = '/home/graysonpetter/extssd/subsample_sats/'

def write_pickle(filename, pickleobj):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(pickleobj, f)

def read_pickle(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)

def reset_params(ball):
    ball.tracers['QSO']['logM1'] = 16.
    ball.tracers['QSO']['sigma'] = 1e-4
    ball.tracers['QSO']['logM_cut'] = 12.2
    ball.tracers['QSO']['Bcent'] = 0.
    ball.tracers['QSO']['alpha_c'] = 0.
    ball.tracers['QSO']['alpha_s'] = 0.
    ball.tracers['QSO']['s'] = 0.
    ball.tracers['QSO']['s_v'] = 0.
    ball.tracers['QSO']['Bsat'] = 0.
    return ball

def load_sim(z, cent_only, sat_only, nslabs=34, small=False,):
    sim_params['sim_name'] = 'AbacusSummit_base_c000_ph000'
    if small:
        sim_params['sim_name'] = 'AbacusSummit_small_c000_ph3000'
    sim_params['z_mock'] = z
    if cent_only:
        sim_params['subsample_dir'] = '/home/graysonpetter/extssd/subsample_noparticles/'
    elif sat_only:
        sim_params['subsample_dir'] = '/home/graysonpetter/extssd/subsample_sats/'
    else:
        sim_params['subsample_dir'] = '/home/graysonpetter/extssd/subsample_default/'

    n_chunks = floor(34/nslabs)
    ball = AbacusHOD(sim_params, HOD_params, clustering_params, chunk=0, n_chunks=n_chunks, delta_z_dist=qsoz['sigZ'])




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

def min_scale(maxz):
    from colossus.cosmology import cosmology
    cosmo = cosmology.setCosmology('planck18')
    fibercollision = 62. / 206265.
    return cosmo.comovingDistance(0, maxz) * fibercollision