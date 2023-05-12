import yaml
import os
from abacusnbody.hod import prepare_sim
config = yaml.safe_load(open('abacus_hod.yaml'))
config['HOD_params']['want_ranks'] = True
config['HOD_params']['want_rsd'] = True
config['HOD_params']['want_AB'] = True

nthread = 1
config['prepare_sim']['Nparallel_load'] = nthread
#zs = [0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0]
#zs = [1.7, 2.0, 2.5, 3.0]
#zs = [1.1, 1.4]
#zs = [1.1, 1.4, 1.7, 2.0, 2.5, 3.0]
#zs = [2.0, 2.5, 3.0]

def prep_defaults(zs):
	for z in zs:
		config['sim_params']['z_mock'] = z
		config['sim_params']['subsample_dir'] = '/home/graysonpetter/extssd/subsample_default/'

		wantlrg = False
		if z == 0.8:
			wantlrg = True
		config['HOD_params']['tracer_flags']['LRG'] = wantlrg


		with open('abacus_hod.yaml', 'w') as yaml_file:
			yaml_file.write( yaml.dump(config, default_flow_style=False))
		prepare_sim.main('abacus_hod.yaml')
		

# don't keep any particles
def prep_centrals(zs):
	for z in zs:
		config['sim_params']['z_mock'] = z
		config['sim_params']['subsample_dir'] = '/home/graysonpetter/extssd/subsample_noparticles/'
		
		wantlrg = False
		if z == 0.8:
			wantlrg = True
		config['HOD_params']['tracer_flags']['LRG'] = wantlrg
		print(config)

		
		with open('abacus_hod.yaml', 'w') as yaml_file:
			yaml_file.write( yaml.dump(config, default_flow_style=False))
		prepare_sim.main('abacus_hod.yaml', force_centrals=True, force_satellites=False)
		

# don't keep any particles
def prep_satellites(zs):
	for z in zs:
		config['sim_params']['z_mock'] = z
		config['sim_params']['subsample_dir'] = '/home/graysonpetter/extssd/subsample_sats/'

		
		wantlrg = False
		if z == 0.8:
			wantlrg = True
		config['HOD_params']['tracer_flags']['LRG'] = wantlrg

		
		with open('abacus_hod.yaml', 'w') as yaml_file:
			yaml_file.write( yaml.dump(config, default_flow_style=False))
		prepare_sim.main('abacus_hod.yaml', force_centrals=False, force_satellites=True)

def reduce_small_boxes():
	# velocity fields only available in small boxes at z <= 1.4
	zs = [1.1, 1.4]
	config['sim_params']['sim_name'] = 'AbacusSummit_small_c000_ph3000'
	prep_centrals(zs)
	prep_satellites(zs)

reduce_small_boxes()