import os
import warnings

from typing import List, Dict

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from absl import app
from absl import logging
from absl import flags

from banditpylib.bandits import Bandit, OrdinaryMNLBandit, CvarReward, \
    MeanReward
from banditpylib.protocols import Protocol
from banditpylib.learners.ordinary_mnl_learner import Learner, UCB, \
    ThompsonSampling

logging.set_verbosity(logging.INFO)
warnings.simplefilter('ignore')
sns.set(style='darkgrid')

FLAGS = flags.FLAGS
flags.DEFINE_string('params_filename', 'params.json',
                    'file to read input parameters')
flags.DEFINE_string('real_params_filename', 'real_params.json',
                    'file to read input parameters')
flags.DEFINE_string('output_filename', 'data.out',
                    'file to dump generated data')
flags.DEFINE_string('figure_filename', 'fig.pdf', 'figure file name')
flags.DEFINE_boolean('random_params', False,
                     'generate random input parameters')
flags.DEFINE_boolean('data', False, 'generate the data')
flags.DEFINE_boolean('fig', False, 'generate the figure')
flags.DEFINE_boolean('final', False, 'generate the final figure')

flags.DEFINE_boolean('cvar_data', False, 'generate data with cvar info')
flags.DEFINE_boolean('cvar_fig', False, 'generate figure where y axis is cvar')
flags.DEFINE_string('output_cvar_filename', 'cvar_data.out',
                    'file to dump generated distribution data')
flags.DEFINE_string('cvar_figure_filename', 'cvar_fig.pdf',
                    'distribution figure file name')

flags.DEFINE_integer('trials', 5, 'number of repetitions of the game')
flags.DEFINE_integer('processes', 5, 'maximum number of processes to use')
flags.DEFINE_integer('horizon', 10000, 'horizon of the game')
flags.DEFINE_integer('freq', 100, 'frequency to record intermediate rewards')
flags.DEFINE_integer('card_limit', 4, 'cardinality constraint')
flags.DEFINE_integer('percentile', 50, 'percentile of cvar')
# for data when generating random parameters
flags.DEFINE_integer('product_num', 10, 'number of products')
# for cvar data
flags.DEFINE_integer('random_neighbors', 10, 'times of local search')


class SinglePlayerProtocol(Protocol):
  """Single player protocol

  This protocol is used to simulate the game when the learner only has one
  player and the learner only interacts with one bandit environment.
  """
  def __init__(self,
               bandit: Bandit,
               learners: List[Learner],
               intermediate_regrets=None,
               percentile=None):
    """
    Args:
      bandit: bandit environment
      learners: learners to be compared with
      intermediate_regrets: whether to record intermediate regrets
      percentile:
    """
    super().__init__(bandit=bandit, learners=learners)
    self.__intermediate_regrets = \
        intermediate_regrets if intermediate_regrets else []
    self.__percentile = percentile

  @property
  def name(self):
    return 'single_player_protocol'

  # measure the cvar of the stochastic rewards between intermedaite regrets
  def measure(self, stochasitc_rewards: np.ndarray) -> float:
    if len(stochasitc_rewards) == 0:
      return 0
    var = np.percentile(stochasitc_rewards, int(self.__percentile))
    return stochasitc_rewards[stochasitc_rewards <= var].mean()

  def _one_trial(self, random_seed: int, debug: bool) -> Dict:
    if debug:
      logging.set_verbosity(logging.DEBUG)
    np.random.seed(random_seed)

    stochastic_rewards = []

    # reset the bandit environment and the learner
    self.bandit.reset()
    self.current_learner.reset()

    one_trial_data = []
    # number of rounds to communicate with the bandit environment
    adaptive_rounds = 0
    # total actions executed by the bandit environment
    total_actions = 0

    def record_data():
      if self.__percentile:
        one_trial_data.append(
            dict({
                'bandit': self.bandit.name,
                'learner': self.current_learner.name,
                'rounds': adaptive_rounds,
                'total_actions': total_actions,
                'measure': self.measure(np.array(stochastic_rewards)),
                'regret': self.bandit.regret(self.current_learner.goal)
            }))
      else:
        one_trial_data.append(
            dict({
                'bandit': self.bandit.name,
                'learner': self.current_learner.name,
                'rounds': adaptive_rounds,
                'total_actions': total_actions,
                'regret': self.bandit.regret(self.current_learner.goal)
            }))

    while True:
      context = self.bandit.context()
      actions = self.current_learner.actions(context)

      # stop the game if actions returned by the learner are None
      if not actions:
        break

      # record intermediate regrets
      if adaptive_rounds in self.__intermediate_regrets:
        record_data()
        # clear stochastic rewards after each intermediate regret
        stochastic_rewards = []

      feedback = self.bandit.feed(actions)
      for (rewards, _) in feedback:
        stochastic_rewards.extend(list(rewards))
      self.current_learner.update(feedback)

      # information update
      for (_, times) in actions:
        total_actions += times
      adaptive_rounds += 1

    # record final regret
    record_data()
    return one_trial_data


def generate_data(params_filename,
                  data_filename,
                  horizon,
                  card_limit,
                  trials,
                  freq,
                  percentile,
                  processes):
  if os.path.exists(data_filename):
    logging.fatal('%s is not empty!' % data_filename)

  intermediate_regrets = list(range(0, horizon + 1, freq))

  reward = CvarReward(percentile / 100)

  if not os.path.exists(params_filename):
    logging.fatal('%s does not exist!' % params_filename)
  logging.info('reading params from %s.' % params_filename)

  with open(params_filename) as f:
    input_params = json.load(f)
  preference_params = np.array(input_params['preference_params'])
  revenues = np.array(input_params['revenues'])

  bandit = OrdinaryMNLBandit(preference_params=preference_params,
                             revenues=revenues,
                             reward=reward,
                             card_limit=card_limit)
  ucb_bad = UCB(revenues=revenues,
                horizon=horizon,
                reward=MeanReward(),
                name='UCB',
                card_limit=card_limit)
  ts_bad = ThompsonSampling(revenues=revenues,
                            horizon=horizon,
                            reward=MeanReward(),
                            name='TS',
                            card_limit=card_limit)
  ucb = UCB(revenues=revenues,
            horizon=horizon,
            reward=reward,
            name='RiskAwareUCB',
            card_limit=card_limit)
  ts = ThompsonSampling(revenues=revenues,
                        horizon=horizon,
                        reward=reward,
                        name='RiskAwareTS',
                        card_limit=card_limit)

  learners = [ucb, ts, ucb_bad, ts_bad]

  # create a new file if possible
  with open(data_filename, 'w'):
    pass
  game = SinglePlayerProtocol(bandit=bandit,
                              learners=learners,
                              intermediate_regrets=intermediate_regrets)
  game.play(trials=trials,
            output_filename=data_filename,
            processes=processes)


def generate_data_with_cvar(params_filename,
                            data_filename,
                            horizon,
                            card_limit,
                            trials,
                            freq,
                            percentile,
                            processes,
                            random_neighbors):
  if os.path.exists(data_filename):
    logging.fatal('%s is not empty!' % data_filename)

  intermediate_regrets = list(range(0, horizon + 1, freq))

  reward = CvarReward(percentile / 100)

  if not os.path.exists(params_filename):
    logging.fatal('%s does not exist!' % params_filename)
  logging.info('reading params from %s.' % params_filename)

  # read parameters from file
  with open(params_filename) as f:
    input_params = json.load(f)
  preference_params = np.array(input_params['preference_params'])
  revenues = np.array(input_params['revenues'])

  bandit = OrdinaryMNLBandit(preference_params=preference_params,
                             revenues=revenues,
                             reward=reward,
                             card_limit=card_limit,
                             zero_best_reward=True)
  ucb_bad = UCB(revenues=revenues,
                horizon=horizon,
                reward=MeanReward(),
                name='UCB',
                card_limit=card_limit,
                use_local_search=True,
                random_neighbors=random_neighbors)
  ts_bad = ThompsonSampling(revenues=revenues,
                            horizon=horizon,
                            reward=MeanReward(),
                            name='TS',
                            card_limit=card_limit,
                            use_local_search=True,
                            random_neighbors=random_neighbors)
  ucb = UCB(revenues=revenues,
            horizon=horizon,
            reward=reward,
            name='RiskAwareUCB',
            card_limit=card_limit,
            use_local_search=True,
            random_neighbors=random_neighbors)
  ts = ThompsonSampling(revenues=revenues,
                        horizon=horizon,
                        reward=reward,
                        name='RiskAwareTS',
                        card_limit=card_limit,
                        use_local_search=True,
                        random_neighbors=random_neighbors)

  learners = [ucb_bad, ts_bad, ts, ucb]

  # create a new file if possible
  with open(data_filename, 'w'):
    pass

  game = SinglePlayerProtocol(bandit=bandit,
                              learners=learners,
                              intermediate_regrets=intermediate_regrets,
                              percentile=percentile)
  game.play(trials=trials,
            output_filename=data_filename,
            processes=processes,
            debug=False)


def make_figure_using_cvar(data_filename, figure_filename):
  with open(data_filename, 'r') as f:
    trials = []
    lines = f.readlines()
    for line in lines:
      trials.append(json.loads(line))
    data_df = pd.DataFrame.from_dict(trials)
  sns.lineplot(x='total_actions', y='measure', hue='learner', data=data_df)
  plt.xlabel(r'$t$', fontweight='bold', fontsize=16)
  plt.ylabel(r'$\mathrm{CVaR}_{0.05}$', fontweight='bold', fontsize=16)
  plt.legend(loc=4)
  plt.savefig(figure_filename, format='pdf')


def make_figure(data_filename, figure_filename):
  with open(data_filename, 'r') as f:
    trials = []
    lines = f.readlines()
    for line in lines:
      trials.append(json.loads(line))
    data_df = pd.DataFrame.from_dict(trials)
  sns.lineplot(x='total_actions', y='regret', hue='learner', data=data_df)
  plt.savefig(figure_filename, format='pdf')


def make_figure_with_worst_regret():
  data_df = pd.DataFrame(columns=['learner', 'total_actions', 'regret'])
  for filename in os.listdir(os.path.join(os.getcwd(), 'arxiv')):
    # read all data files
    if 'data' in filename:
      with open(os.path.join('arxiv', filename), 'r') as f:
        trials = []
        lines = f.readlines()
        for line in lines:
          trials.append(json.loads(line))
        data_df = data_df.append(pd.DataFrame.from_dict(trials)[[
            'learner', 'total_actions', 'regret'
        ]].groupby(['learner', 'total_actions']).mean().reset_index(),
                                 ignore_index=True)
  data_df = data_df.groupby(['learner', 'total_actions']).max().reset_index()

  learners = set(data_df['learner'])
  for learner in learners:
    # from sklearn.linear_model import LinearRegression
    x = np.array(data_df[data_df.learner == learner]['total_actions']).reshape(
        -1, 1)
    y = np.array(data_df[data_df.learner == learner]['regret'])
    # reg = LinearRegression().fit(np.sqrt(x)[10:], y[10:])
    plt.plot(np.sqrt(x), y, label=learner)
    # plt.plot(np.sqrt(x),
    #          reg.coef_[0] * np.sqrt(x) + reg.intercept_,
    #          '--',
    #          label=r'y = %.2f' % reg.coef_[0] + r'$\sqrt{x}$' +
    #                 r' + %.2f' % reg.intercept_)
  plt.xlabel(r'$\sqrt{t}$', fontweight='bold', fontsize=16)
  plt.ylabel('regret', fontweight='bold', fontsize=16)
  plt.legend(loc='upper left')
  plt.savefig('worst_regret.pdf', format='pdf')


def generate_random_params(output_filename, product_num):
  # randomly generate input params
  if os.path.exists(output_filename):
    logging.fatal('%s is not empty!' % output_filename)
  preference_params = [1]
  revenues = [0]
  # pylint: disable=no-member
  preference_params.extend(list(np.random.random(product_num)))
  revenues.extend(list(np.random.uniform(0.1, 1, product_num)))
  params = dict({
      'preference_params': preference_params,
      'revenues': revenues
  })
  with open(output_filename, 'w') as f:
    json.dump(params, f)


def main(args):
  del args

  if FLAGS.random_params:
    generate_random_params(output_filename=FLAGS.params_filename,
                           product_num=FLAGS.product_num)

  if FLAGS.data:
    generate_data(params_filename=FLAGS.params_filename,
                  data_filename=FLAGS.output_filename,
                  horizon=FLAGS.horizon,
                  card_limit=FLAGS.card_limit,
                  trials=FLAGS.trials,
                  freq=FLAGS.freq,
                  percentile=FLAGS.percentile,
                  processes=FLAGS.processes)

  if FLAGS.fig:
    make_figure(data_filename=FLAGS.output_filename,
                figure_filename=FLAGS.figure_filename)

  if FLAGS.final:
    make_figure_with_worst_regret()

  if FLAGS.cvar_data:
    generate_data_with_cvar(params_filename=FLAGS.real_params_filename,
                            data_filename=FLAGS.output_cvar_filename,
                            horizon=FLAGS.horizon,
                            card_limit=FLAGS.card_limit,
                            trials=FLAGS.trials,
                            freq=FLAGS.freq,
                            percentile=FLAGS.percentile,
                            processes=FLAGS.processes,
                            random_neighbors=FLAGS.random_neighbors)

  if FLAGS.cvar_fig:
    make_figure_using_cvar(data_filename=FLAGS.output_cvar_filename,
                           figure_filename=FLAGS.cvar_figure_filename)


if __name__ == '__main__':
  app.run(main)
