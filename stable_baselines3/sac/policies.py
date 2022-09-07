import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import copy

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import polyak_update

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# num of stacked observations
NUM_STACK = 4
# limbs diff upper limit for dynamic id input
UPPER_DIFF_LIMBS = 1.0 * NUM_STACK**0.5

class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class HumarlActor(BasePolicy):
    """
    Actor network (policy) for multi-agent SAC on humanoid env.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)
        if self.use_sde:
            assert False, "do not use sde in humarl"
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        action_dims = [3, 4, 4, 3, 3]
        self.action_dist = SquashedDiagGaussianDistribution(sum(action_dims))

        # self.ind_obses = [
        #     list(range(0,11)) + [12,13,14,16,17,19,20] + list(range(22,34)) + [35,36,37,39,40,42,43]
        #         + list(range(55,95)) + list(range(115,125)) + list(range(145,155)) + list(range(165,175))
        #         + list(range(191,215)) + list(range(227,233)) + list(range(245,251)) + list(range(257,263))
        #         + list(range(275,281)) + [282,283,284,286,287,289,290]
        #         + list(range(298,322)) + list(range(334,340)) + list(range(352,358)) + list(range(364,370)),
        #     list(range(0,12)) + list(range(22,35)) + list(range(75,115)) + list(range(203,227)) + list(range(278,282)) + list(range(310,334)),
        #     list(range(0,8)) + list(range(12,16)) + list(range(22,31)) + list(range(35,39)) + list(range(75,85)) + list(range(115,145))
        #         + list(range(203,209)) + list(range(227,245)) + list(range(282,286)) + list(range(310,316)) + list(range(334,352)),
        #     list(range(0,8)) + [16,17,18] + list(range(22,31)) + [39,40,41] + list(range(55,65)) + list(range(145,165)) + list(range(191,197))
        #         + list(range(245,257)) + [286,287,288] + list(range(298,304)) + list(range(352,364)),
        #     list(range(0,8)) + [19,20,21] + list(range(22,31)) + [42,43,44] + list(range(55,65)) + list(range(165,185)) + list(range(191,197))
        #         + list(range(257,269)) + [289,290,291] + list(range(298,304)) + list(range(364,376)),
        # ] # [204, 117, 117, 92, 92]

        ### truncated observation
        ## local observation for body agents
        # self.ind_obses = [
        #     list(range(0,11)) + [12,13,14,16,17,19,20] + list(range(22,34)) + [35,36,37,39,40,42,43],
        #     list(range(0,12)) + list(range(22,35)),
        #     list(range(0,8)) + list(range(12,16)) + list(range(22,31)) + list(range(35,39)),
        #     list(range(0,8)) + [16,17,18] + list(range(22,31)) + [39,40,41],
        #     list(range(0,8)) + [19,20,21] + list(range(22,31)) + [42,43,44],
        # ] # [37,25,25,23,23]
        ## global observation for body agents
        self.ind_obses = [
            list(range(45)) for _ in action_dims
        ] # [45,45,45,45,45]

        self.ind_inverse_obs = [[] for _ in self.ind_obses]
        self.ind_swap_to_obs = [[] for _ in self.ind_obses]
        self.ind_swap_from_obs = [[] for _ in self.ind_obses]
        self.ind_inverse_act = [[] for _ in self.ind_obses]
        ### projecting observation/action spaces of left leg/arm to those of right leg/arm
        ## local obs
        # TODO the local observation/action projection is wrong, to correct!!!
        # self.ind_inverse_obs[2] = [2,4,5,7,8,9,13,15,17,18,20,21,22]
        # self.ind_inverse_obs[4] = [2,4,5,7,9,12,14,16,17,19,21]
        # self.ind_inverse_act[2] = [0,1]
        # self.ind_inverse_act[4] = [1]
        ## global obs
        self.ind_inverse_obs[2] = [2,4,5,7,16,17,19,20,23,25,27,28,30,39,40,42,43]
        ind_right_leg = list(range(8,12))
        ind_right_leg_v = [i+23 for i in ind_right_leg]
        ind_left_leg = list(range(12,16))
        ind_left_leg_v = [i+23 for i in ind_left_leg]
        ind_right_upper_arm = [16,17]
        ind_right_upper_arm_v = [i+23 for i in ind_right_upper_arm]
        ind_right_elbow = [18,]
        ind_right_elbow_v = [i+23 for i in ind_right_elbow]
        ind_left_upper_arm = [19,20]
        ind_left_upper_arm_v = [i+23 for i in ind_left_upper_arm]
        ind_left_elbow = [21,]
        ind_left_elbow_v = [i+23 for i in ind_left_elbow]
        self.ind_swap_to_obs[2] = ind_right_leg + ind_left_leg + ind_right_upper_arm + ind_right_elbow + ind_left_upper_arm + ind_left_elbow \
                                + ind_right_leg_v + ind_left_leg_v + ind_right_upper_arm_v + ind_right_elbow_v + ind_left_upper_arm_v + ind_left_elbow_v
        self.ind_swap_from_obs[2] = ind_left_leg + ind_right_leg + ind_left_upper_arm + ind_left_elbow + ind_right_upper_arm + ind_right_elbow \
                                + ind_left_leg_v + ind_right_leg_v + ind_left_upper_arm_v + ind_left_elbow_v + ind_right_upper_arm_v + ind_right_elbow_v
        self.ind_inverse_obs[4] = self.ind_inverse_obs[2]
        self.ind_swap_to_obs[4] = self.ind_swap_to_obs[2]
        self.ind_swap_from_obs[4] = self.ind_swap_from_obs[2]
        self.ind_inverse_act[4] = [0,1]

        
        obs_dims = [len(ind) for ind in self.ind_obses]
        ### id input layer
        obs_dims[1:] = [dim+2 for dim in obs_dims[1:]]
        ### dynamic id input layer
        # obs_dims[1:] = [dim+1 for dim in obs_dims[1:]]
        

        ### local obs, latent ps=F
        # latent_pi_nets = [create_mlp(obs_dim, -1, net_arch, activation_fn) for obs_dim in obs_dims]

        ### local obs, latent ps=T
        latent_pi_torso = create_mlp(obs_dims[0], -1, net_arch, activation_fn)
        latent_pi_leg = create_mlp(obs_dims[1], -1, net_arch, activation_fn)
        latent_pi_arm = create_mlp(obs_dims[3], -1, net_arch, activation_fn)
        latent_pi_nets = [latent_pi_torso, latent_pi_leg, latent_pi_leg, latent_pi_arm, latent_pi_arm]
        
        ### last layer ps=F
        # mu_nets = [nn.Linear(last_layer_dim, action_dim) for action_dim in action_dims]
        # log_std_nets = [nn.Linear(last_layer_dim, action_dim) for action_dim in action_dims]

        ### last layer ps=T
        mu_net_leg = nn.Linear(last_layer_dim, action_dims[1])
        mu_net_arm = nn.Linear(last_layer_dim, action_dims[3])
        log_std_net_leg = nn.Linear(last_layer_dim, action_dims[1])
        log_std_net_arm = nn.Linear(last_layer_dim, action_dims[3])
        mu_nets = [nn.Linear(last_layer_dim, action_dims[0]), mu_net_leg, mu_net_leg, mu_net_arm, mu_net_arm]
        log_std_nets = [nn.Linear(last_layer_dim, action_dims[0]), log_std_net_leg, log_std_net_leg, log_std_net_arm, log_std_net_arm]
        
        self.latent_pis = nn.ModuleList([nn.Sequential(*latent_pi_net) for latent_pi_net in latent_pi_nets])
        self.mus = nn.ModuleList(mu_nets)
        self.log_stds = nn.ModuleList(log_std_nets)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        
        latent_pis = []
        ### dynamic id input layer
        ## TODO, optimize the calculation of diff_limbs_obses by removing duplicate items in current calculation
        # diff_limbs_obses = (features[..., self.ind_obses[1]]-features[..., self.ind_obses[2]]).norm(dim=-1)\
        #     + (features[..., self.ind_obses[3]]-features[..., self.ind_obses[4]]).norm(dim=-1)
        # diff_limbs_obses[diff_limbs_obses>UPPER_DIFF_LIMBS] = 1.0
        # id_dynamic = (1.0 - diff_limbs_obses)[..., None]

        for latent_pi,ind_obs,inverse_obs,idx,swap_to_obs,swap_from_obs in zip(self.latent_pis, self.ind_obses, self.ind_inverse_obs, range(5),
                                                                                self.ind_swap_to_obs, self.ind_swap_from_obs):
            input = features[..., ind_obs]
            input[..., inverse_obs] *= -1
            input[..., swap_to_obs] = input[..., swap_from_obs]
            ### id input layer
            if idx in [1,3]:
                rep_shape = list(input.shape)
                rep_shape[-1] = 1
                input = th.cat((th.tensor([1,0],device=obs.device).repeat(rep_shape), input), dim=-1)
            elif idx in [2,4]:
                rep_shape = list(input.shape)
                rep_shape[-1] = 1
                input = th.cat((th.tensor([0,1],device=obs.device).repeat(rep_shape), input), dim=-1)

            ### dynamic id input layer
            # if idx in [1,3]:
            #     input = th.cat((id_dynamic, input), dim=-1)
            # elif idx in [2,4]:
            #     input = th.cat((-id_dynamic, input), dim=-1)

            latent = latent_pi(input)

            latent_pis.append(latent)

        mean_actions_list = []
        for latent_pi,mu,inverse_act in zip(latent_pis, self.mus, self.ind_inverse_act):
            output_mu = mu(latent_pi)
            output_mu[..., inverse_act] *= -1
            mean_actions_list.append(output_mu)

        if self.use_sde:
            assert False, "do not use sde in humarl"
        # Unstructured exploration (Original implementation)
        log_std_list = [log_std(latent_pi) for log_std,latent_pi in zip(self.log_stds, latent_pis)]
        # Original Implementation to cap the standard deviation
        log_std_list = [th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) for log_std in log_std_list]

        mean_actions = th.cat(mean_actions_list, dim=1)
        log_std = th.cat(log_std_list, dim=1)

        return mean_actions, log_std, {}

    def reset_lazy(self, tau: float):
        models_reset = [self.latent_pis[3][0], self.latent_pis[3][2], self.mus[3], self.log_stds[3]]
        models_init = copy.deepcopy(models_reset)
        for model_reset, model_init in zip(models_reset, models_init):
            model_init.reset_parameters()
            polyak_update(model_init.parameters(), model_reset.parameters(), tau)

        # # reset actors for arms
        # self.latent_pis[3][0].reset_parameters()
        # self.latent_pis[3][2].reset_parameters()
        # self.mus[3].reset_parameters()
        # # self.mus[4].reset_parameters()
        # self.log_stds[3].reset_parameters()
        # # self.log_stds[4].reset_parameters()

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class WalkerActor(BasePolicy):
    """
    Actor network (policy) for multi-agent SAC on Walker2d env.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)
        if self.use_sde:
            assert False, "do not use sde in humarl"
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        action_dims = [3, 3]
        self.action_dist = SquashedDiagGaussianDistribution(sum(action_dims))

        # global observation for body agents
        obs_left = [0,1,5,6,7,2,3,4,8,9,10,14,15,16,11,12,13]
        obs_left_stack = [[i+17*n for i in obs_left] for n in range(NUM_STACK)]
        self.ind_obses = [
            list(range(17*NUM_STACK)),
            [element for sublist in obs_left_stack for element in sublist]
        ] # [17*NUM_STACK,17*NUM_STACK]
        
        obs_dims = [len(ind) for ind in self.ind_obses]
        ### id input layer
        # obs_dims = [dim+2 for dim in obs_dims]
        ### dynamic id input layer
        # obs_dims = [dim+1 for dim in obs_dims]
        ### id thigh forward
        obs_dims = [dim+2 for dim in obs_dims]
        

        ### local obs, latent ps=F
        # latent_pi_nets = [create_mlp(obs_dim, -1, net_arch, activation_fn) for obs_dim in obs_dims]

        ### local obs, latent ps=T
        latent_pi_leg = create_mlp(obs_dims[0], -1, net_arch, activation_fn)
        latent_pi_nets = [latent_pi_leg, latent_pi_leg]
        
        ### last layer ps=F
        # mu_nets = [nn.Linear(last_layer_dim, action_dim) for action_dim in action_dims]
        # log_std_nets = [nn.Linear(last_layer_dim, action_dim) for action_dim in action_dims]

        ### last layer ps=T
        mu_net_leg = nn.Linear(last_layer_dim, action_dims[1])
        log_std_net_leg = nn.Linear(last_layer_dim, action_dims[1])
        mu_nets = [mu_net_leg, mu_net_leg]
        log_std_nets = [log_std_net_leg, log_std_net_leg]
        
        self.latent_pis = nn.ModuleList([nn.Sequential(*latent_pi_net) for latent_pi_net in latent_pi_nets])
        self.mus = nn.ModuleList(mu_nets)
        self.log_stds = nn.ModuleList(log_std_nets)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        
        latent_pis = []

        ### no id
        # latent_pis = [latent_pi(features[...,ind_obs]) for latent_pi,ind_obs in zip(self.latent_pis, self.ind_obses)]
        ### id input layer
        # ids = th.eye(2, device=features.device)
        # latent_pis = [latent_pi(th.cat((ids[th.ones_like(features[...,0],dtype=th.long)*ind], features[..., ind_obs]), dim=-1))
        #                 for latent_pi,ind_obs,ind in zip(self.latent_pis, self.ind_obses, range(2))]
        ### dynamic id input layer
        ## TODO, optimize the calculation of diff_limbs_obses by removing duplicate items in current calculation
        # diff_limbs_obses = (features[..., self.ind_obses[0]]-features[..., self.ind_obses[1]]).norm(dim=-1)
        # diff_limbs_obses[diff_limbs_obses>UPPER_DIFF_LIMBS] = UPPER_DIFF_LIMBS
        # id_dynamic = (1.0 - diff_limbs_obses/UPPER_DIFF_LIMBS)[..., None]
        # latent_pis = [latent_pi(th.cat((id_dynamic*sign, features[..., ind_obs]), dim=-1))
        #                 for latent_pi,ind_obs,sign in zip(self.latent_pis, self.ind_obses, [1,-1])]
        ### id thigh forward
        ids = th.eye(2, device=features.device)
        right_thigh_forward = (features[..., -6] - features[..., -3]) > 0
        latent_pis = [self.latent_pis[0](th.cat((ids[right_thigh_forward.long()], features[..., self.ind_obses[0]]), dim=-1)),
                        self.latent_pis[1](th.cat((ids[(~right_thigh_forward).long()], features[..., self.ind_obses[1]]), dim=-1))]

        mean_actions_list = [mu(latent_pi) for mu,latent_pi in zip(self.mus, latent_pis)]

        if self.use_sde:
            assert False, "do not use sde in humarl"
        # Unstructured exploration (Original implementation)
        log_std_list = [log_std(latent_pi) for log_std,latent_pi in zip(self.log_stds, latent_pis)]
        # Original Implementation to cap the standard deviation
        log_std_list = [th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) for log_std in log_std_list]

        mean_actions = th.cat(mean_actions_list, dim=1)
        log_std = th.cat(log_std_list, dim=1)

        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class SACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = net_arch, [256, 256] # get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = SACPolicy


class CnnPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class MultiInputPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class HumarlPolicy(SACPolicy):
    """
    Multi-agent policy class (with both actor and critic) for SAC on environment Humanoid.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        local_obs: bool = False,
        ps: bool = False,
    ):
        self.local_obs = local_obs
        self.ps = ps
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        if self.observation_space.shape[0] == 376*NUM_STACK:
            return HumarlActor(**actor_kwargs).to(self.device)
        elif self.observation_space.shape[0] == 17*NUM_STACK:
            return WalkerActor(**actor_kwargs).to(self.device)