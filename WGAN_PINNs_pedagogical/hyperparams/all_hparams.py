from utils import registry, hparam

"""
Test for different top_k [1, 10, 20, 30, 40, 50]
"""


@registry.register_hparam('basic_params1')
def basic_params1():
    return hparam.HParam(noise_level=0.05,
                         N_r=100,
                         N_u=20,
                         par_pinns=1e2,
                         data_shape=1,
                         z_shape=1,
                         num_itr=30000,
                         g_depth=3,
                         g_width=50,
                         d_depth=3,
                         d_width=50,
                         lrg=1e-4,
                         lrd=1e-4,
                         beta_1=0.9,
                         beta_2=0.99,
                         bjorck_beta=0.5,
                         bjorck_iter=5,
                         bjorck_order=2,
                         group_size=2
                         )





@registry.register_hparam('PDE')
def movie_len_1m_params(basic_param_name):
    basic_param_fn = registry.get_hparam(basic_param_name)
    basic_param = basic_param_fn()

    assert isinstance(basic_param, hparam.HParam)

    basic_param.add_params(dataset_name='pde'
                           )

    return basic_param
