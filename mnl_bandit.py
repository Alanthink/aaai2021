import os
import warnings

from typing import List

import json
# To avoid type 3 fonts
# See http://phyletica.org/matplotlib-fonts/ for more details.
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from absl import app
from absl import logging
from absl import flags

from banditpylib.bandits import Bandit, MNLBandit, CvarReward, \
    MeanReward
from banditpylib.data_pb2 import Trial
from banditpylib.protocols import Protocol, trial_data_messages_to_dict
from banditpylib.learners.mnl_bandit_learner import Learner, UCB, \
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

FONT_SIZE = 16


class SinglePlayerProtocol(Protocol):
  """Single player protocol

  This protocol is used to simulate the game when the learner only has one
  player and the learner only interacts with one bandit environment.
  """
  def __init__(self,
               bandit: Bandit,
               learners: List[Learner],
               intermediate_regrets=None,
               percentile=None,
               horizon: int = np.inf): # type: ignore
    """
    Args:
      bandit: bandit environment
      learners: learners to be compared with
      intermediate_regrets: whether to record intermediate regrets
      percentile: cvar percentile
      horizon: horizon of the game (i.e., total number of actions a leaner can
        make)
    """
    super().__init__(bandit=bandit, learners=learners)
    self.__intermediate_regrets = \
        intermediate_regrets if intermediate_regrets else []
    self.__percentile = percentile
    self.__horizon = horizon

  @property
  def name(self):
    return 'single_player_protocol'

  # measure the cvar of the stochastic rewards between intermedaite regrets
  def __measure(self, stochasitc_rewards: np.ndarray) -> float:
    if len(stochasitc_rewards) == 0:
      return 0
    var = np.percentile(stochasitc_rewards, int(self.__percentile))
    return stochasitc_rewards[stochasitc_rewards <= var].mean()

  def _one_trial(self, random_seed: int, debug: bool) -> bytes:
    if debug:
      logging.set_verbosity(logging.DEBUG)
    np.random.seed(random_seed)

    stochastic_rewards = []

    # reset the bandit environment and the learner
    self.bandit.reset()
    self.current_learner.reset()

    trial = Trial()
    trial.bandit = self.bandit.name
    trial.learner = self.current_learner.name
    # number of rounds to communicate with the bandit environment
    rounds = 0
    # total actions executed by the bandit environment
    total_actions = 0

    def add_data():
      data_item = trial.data_items.add()
      if self.__percentile:
        data_item.other = self.__measure(np.array(stochastic_rewards))
      data_item.rounds = rounds
      data_item.total_actions = total_actions
      data_item.regret = self.bandit.regret(self.current_learner.goal)

    while total_actions < self.__horizon:
      actions = self.current_learner.actions(self.bandit.context)

      # stop the game if no actions are returned by the learner
      if not actions.arm_pulls:
        break

      # record intermediate regrets
      if rounds in self.__intermediate_regrets:
        add_data()
        # clear stochastic rewards after each intermediate regret
        stochastic_rewards = []

      feedback = self.bandit.feed(actions)
      for arm_feedback in feedback.arm_feedbacks:
        stochastic_rewards.extend(arm_feedback.rewards)
      self.current_learner.update(feedback)

      # information update
      for arm_pull in actions.arm_pulls:
        total_actions += arm_pull.times
      rounds += 1

    # record final regret
    add_data()
    return trial.SerializeToString()


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

  bandit = MNLBandit(preference_params=preference_params,
                             revenues=revenues,
                             reward=reward,
                             card_limit=card_limit)
  ucb_bad = UCB(revenues=revenues,
                reward=MeanReward(),
                name='UCB',
                card_limit=card_limit)
  ts_bad = ThompsonSampling(revenues=revenues,
                            horizon=horizon,
                            reward=MeanReward(),
                            name='TS',
                            card_limit=card_limit)
  ucb = UCB(revenues=revenues,
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
                              intermediate_regrets=intermediate_regrets,
                              horizon=horizon)
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

  bandit = MNLBandit(preference_params=preference_params,
                             revenues=revenues,
                             reward=reward,
                             card_limit=card_limit,
                             zero_best_reward=True)
  ucb_bad = UCB(revenues=revenues,
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
                              percentile=percentile,
                              horizon=horizon)
  game.play(trials=trials,
            output_filename=data_filename,
            processes=processes,
            debug=False)


def make_figure_using_cvar(data_filename, figure_filename):
  data_df = trial_data_messages_to_dict(data_filename)
  ax = sns.lineplot(x='total_actions', y='other', hue='learner', data=data_df)
  ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
  plt.xlabel(r'$t$', fontweight='bold', fontsize=FONT_SIZE)
  plt.ylabel(r'$\mathrm{CVaR}_{0.05}$', fontweight='bold', fontsize=FONT_SIZE)
  plt.xticks(fontsize=FONT_SIZE)
  plt.yticks(fontsize=FONT_SIZE)
  plt.legend(loc=4, fontsize=FONT_SIZE)
  plt.savefig(figure_filename, format='pdf', bbox_inches = 'tight')


def make_figure(data_filename, figure_filename):
  data_df = trial_data_messages_to_dict(data_filename)
  sns.lineplot(x='total_actions', y='regret', hue='learner', data=data_df)
  plt.savefig(figure_filename, format='pdf')


def make_figure_with_worst_regret():
  data_df = pd.DataFrame(columns=['learner', 'total_actions', 'regret'])
  for filename in os.listdir(os.path.join(os.getcwd(), 'arxiv')):
    # read all data files
    if 'data' in filename:
      trials = trial_data_messages_to_dict(os.path.join('arxiv', filename))
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
  plt.xticks(fontsize=FONT_SIZE)
  plt.yticks(fontsize=FONT_SIZE)
  plt.xlabel(r'$\sqrt{t}$', fontweight='bold', fontsize=FONT_SIZE)
  plt.ylabel('regret', fontweight='bold', fontsize=FONT_SIZE)
  plt.legend(loc='upper left', fontsize=FONT_SIZE)
  plt.savefig('worst_regret.pdf', format='pdf', bbox_inches = 'tight')


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
