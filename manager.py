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