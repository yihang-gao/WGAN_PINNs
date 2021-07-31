from utils import problem, registry
from models import BaseGAN
from utils.loaddata import get_data


@registry.register_problem('GAN')
class GANProblem(problem.Problem):
    def __init__(self, hparam):
        super().__init__(hparam)

    def load_data(self):
        self.XYU_u, self.XY_r, self.XY_test, self.X_mean, self.X_std, self.Y_mean, self.Y_std = get_data(noise_level=self.hparam['noise_level'],
                                                                                N_r=self.hparam['N_r'],
                                                                                N_u=self.hparam['N_u'])

    def train_model(self):
        gan = BaseGAN(
            noise_level=self.hparam['noise_level'],
            N_r=self.hparam['N_r'],
            N_u=self.hparam['N_u'],
            X_mean=self.X_mean,
            X_std=self.X_std,
            Y_mean=self.Y_mean,
            Y_std=self.Y_std,
            par_pinns=self.hparam['par_pinns'],
            z_shape=self.hparam['z_shape'],
            out_dim=self.hparam['data_shape'],
            num_itr=self.hparam['num_itr'],
            g_depth=self.hparam['g_depth'],
            g_width=self.hparam['g_width'],
            d_depth=self.hparam['d_depth'],
            d_width=self.hparam['d_width'],
            lrg=self.hparam['lrg'],
            lrd=self.hparam['lrd'],
            beta_1=self.hparam['beta_1'],
            beta_2=self.hparam['beta_2'],
            bjorck_beta=self.hparam['bjorck_beta'],
            bjorck_iter=self.hparam['bjorck_iter'],
            bjorck_order=self.hparam['bjorck_order'],
            group_size=self.hparam['group_size']
        )
        '''
        R2sq, L2sq = gan.train(self.XY_u, self.X_r, self.X_test)
        name_file = "/PDEWGAN/stat/harapinns_{:.2f}_noise_{:.2f}_Nr_{}_Nu_{}".format(self.hparam['par_pinns'], self.hparam['noise_level'], self.hparam['N_r'], self.hparam['N_u'])
        return R2sq, L2sq, name_file
        '''
        gan.train(self.XYU_u, self.XY_r, self.XY_test)

    def test_model(self):
        pass
