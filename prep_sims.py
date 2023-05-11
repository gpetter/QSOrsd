import yaml
import os
from abacusnbody.hod import prepare_sim
path2config = '/home/graysonpetter/ssd/Dartmouth/data/simulations/.config/abacus_hod.yaml'
config = yaml.safe_load(open(path2config))

nthread = 2

#zs = [0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0]
#zs = [1.7, 2.0, 2.5, 3.0]
#zs = [1.1, 1.4]
#zs = [1.1, 1.4, 1.7, 2.0, 2.5, 3.0]
zs = [2.0, 2.5, 3.0]

def prep_defaults():
	for z in zs:
		config['sim_params']['z_mock'] = z
		config['sim_params']['subsample_dir'] = '/home/graysonpetter/extssd/subsample_default/'
		config['prepare_sim']['Nparallel_load'] = nthread
		
		wantlrg = False
		if z == 0.8:
			wantlrg = True
		config['HOD_params']['tracer_flags']['LRG'] = wantlrg
		config['HOD_params']['want_ranks'] = True
		config['HOD_params']['want_rsd'] = True
		config['HOD_params']['want_AB'] = True
		
		with open('abacus_hod.yaml', 'w') as yaml_file:
			yaml_file.write( yaml.dump(config, default_flow_style=False))
		prepare_sim.main('abacus_hod.yaml')
		

# don't keep any particles
def prep_centrals():
	for z in zs:
		config['sim_params']['z_mock'] = z
		config['sim_params']['subsample_dir'] = '/home/graysonpetter/extssd/subsample_noparticles/'
		config['prepare_sim']['Nparallel_load'] = nthread
		
		wantlrg = False
		if z == 0.8:
			wantlrg = True
		config['HOD_params']['tracer_flags']['LRG'] = wantlrg
		config['HOD_params']['want_ranks'] = True
		config['HOD_params']['want_rsd'] = True
		config['HOD_params']['want_AB'] = True
		
		with open('abacus_hod.yaml', 'w') as yaml_file:
			yaml_file.write( yaml.dump(config, default_flow_style=False))
		prepare_sim.main('abacus_hod.yaml')
		

# don't keep any particles
def prep_satellites():
	for z in zs:
		config['sim_params']['z_mock'] = z
		config['sim_params']['subsample_dir'] = '/home/graysonpetter/extssd/subsample_sats/'
		config['prepare_sim']['Nparallel_load'] = nthread
		
		wantlrg = False
		if z == 0.8:
			wantlrg = True
		config['HOD_params']['tracer_flags']['LRG'] = wantlrg
		config['HOD_params']['want_ranks'] = True
		config['HOD_params']['want_rsd'] = True
		config['HOD_params']['want_AB'] = True
		
		with open('abacus_hod.yaml', 'w') as yaml_file:
			yaml_file.write( yaml.dump(config, default_flow_style=False))
		prepare_sim.main('abacus_hod.yaml')

prep_satellites()