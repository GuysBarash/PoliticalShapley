import time

import numpy as np
import pandas as pd
from IPython.display import display
import itertools
import os
import shutil
from tqdm import tqdm
from copy import deepcopy
from tabulate import tabulate

from PoliticalShapley import clear_folder
from PoliticalShapley import PoliticalShapley


class StateDelta:
    def __init__(self, old_state, new_state, old_gov=None, new_gov=None):
        self.old_state = old_state
        self.new_state = new_state
        self.old_gov = old_gov
        self.old_opposition = None
        if old_gov is not None:
            self.old_opposition = [k for k in self.old_state.parties.keys() if k not in self.old_gov]
        self.new_gov = new_gov
        self.new_opposition = None
        if new_gov is not None:
            self.new_opposition = [k for k in self.new_state.parties.keys() if k not in self.new_gov]

        self.parties = list(set(list(self.old_state.parties.keys()) + list(self.new_state.parties.keys())))
        self.seatsdf = pd.DataFrame(
            columns=['Old seats', 'New seats', 'Delta'],
            index=self.parties,
            dtype=int,
        )
        self.seatsdf.loc[self.old_state.parties.keys(), 'Old seats'] = self.old_state.get_mandates()['Mandates']
        self.seatsdf.loc[self.new_state.parties.keys(), 'New seats'] = self.new_state.get_mandates()['Mandates']
        self.seatsdf = self.seatsdf.fillna(0).astype(int)
        self.seatsdf['Delta'] = self.seatsdf['New seats'] - self.seatsdf['Old seats']
        self.seatsdf = self.seatsdf.sort_values(by='Delta', ascending=False)

        self.shapdf = pd.DataFrame(
            columns=['Old shap', 'New shap', 'Delta'],
            index=self.parties,
            dtype=float,
        )
        self.shapdf.loc[self.old_state.parties.keys(), 'Old shap'] = self.old_state.get_shapley()
        self.shapdf.loc[self.new_state.parties.keys(), 'New shap'] = self.new_state.get_shapley()
        self.shapdf = self.shapdf.fillna(0.0).astype(float)
        self.shapdf['Delta'] = self.shapdf['New shap'] - self.shapdf['Old shap']
        self.shapdf = self.shapdf.sort_values(by='Delta', ascending=False)

    def zero_opposition_values(self, zero_old_state=True, zero_new_state=True):
        if zero_old_state and self.old_gov is not None:
            self.shapdf.loc[self.old_opposition, 'Old shap'] = 0.0
        if zero_new_state and self.new_gov is not None:
            self.shapdf.loc[self.new_opposition, 'New shap'] = 0.0
        self.shapdf['Delta'] = self.shapdf['New shap'] - self.shapdf['Old shap']
        self.shapdf = self.shapdf.sort_values(by='Delta', ascending=False)


class State(PoliticalShapley):
    def __init__(self, gov=None, campaign_value=1, impossible_moves=None):
        PoliticalShapley.__init__(self)
        self.gov = gov
        self.campaign_value = campaign_value
        self.impossible_moves = impossible_moves
        self.opposition = None
        if self.gov is not None:
            self.opposition = [k for k in self.parties.keys() if k not in self.gov]

    def get_players(self):
        return list(self.parties.keys())

    def get_restrictions(self, player):
        lret = deepcopy(self.disagree.get(player, list()))
        for k in self.disagree.keys():
            if k == player:
                continue
            else:
                if player in self.disagree[k]:
                    lret += [k]

        return list(set(lret))

    def get_score(self, player=None):
        if player is not None:
            shaps = self.get_shapley(player)
            mandates = self.get_mandates(player)
        else:
            shaps = self.get_shapley()
            mandates = self.get_mandates()
        return shaps

    def get_actions(self, player):
        if player not in self.get_players():
            # raise Exception(f"No such player: {player}")
            return list()

        ret = list()
        for trgt in self.get_players():
            if trgt == player:
                continue
            else:
                if self.parties.get(trgt, 0) > 0:
                    t_action = ('Campaign', player, trgt)
                    ret.append(t_action)

        restrictions = self.get_restrictions(player)
        for adv in restrictions:
            t_action = ('Break', player, adv)
            ret.append(t_action)

        if self.impossible_moves is not None:
            ret = [t for t in ret if t not in self.impossible_moves]
        return ret

    def apply(self, action, campaign_power=None):
        action_type, src_player, trgt_player = action

        parties = deepcopy(self.parties)
        disagree = deepcopy(self.disagree)

        if action_type == 'Campaign':
            campaign_value = min(self.campaign_value, parties[trgt_player])
            if campaign_power is not None:
                campaign_value = campaign_power
            parties[src_player] += campaign_value
            parties[trgt_player] -= campaign_value
        if action_type == 'Break':
            if src_player in disagree.get(trgt_player, list()):
                disagree[trgt_player].remove(src_player)
            if trgt_player in disagree.get(src_player, list()):
                disagree[src_player].remove(trgt_player)

        ret_state = State(campaign_value=self.campaign_value, impossible_moves=self.impossible_moves)
        ret_state.add_parties(parties)
        ret_state.add_restrictions(disagree)
        ret_state.run()
        return ret_state

    def copy(self):
        return deepcopy(self)


def get_next_states(state, player):
    actions = state.get_actions(player)
    resdf = pd.DataFrame(
        columns=['idx', 'Action', 'src', 'trgt',
                 'src_gain_prev', 'src gain',
                 'trgt_gain_prev', 'trgt gain',
                 ],
        index=range(len(actions)))
    for idx, action in enumerate(actions):
        tt_state = state.apply(action)
        src_score = tt_state.get_score(player)
        trgt_player = action[2]
        trgt_score = tt_state.get_score(trgt_player)
        resdf.loc[idx, ['idx', 'Action', 'src', 'trgt', 'src gain', 'trgt gain']] = [idx, action[0], action[1],
                                                                                     trgt_player,
                                                                                     src_score, trgt_score]
        resdf.loc[idx, ['src_gain_prev', 'trgt_gain_prev']] = [state.get_score(player), state.get_score(trgt_player)]

    gains = 1
    resdf['src gain delta'] = (resdf['src gain'] - resdf['src_gain_prev']).fillna(0)
    resdf['trgt gain delta'] = (resdf['trgt gain'] - resdf['trgt_gain_prev']).fillna(0)
    resdf = resdf.sort_values(by='src gain delta', ascending=False)
    return resdf, [actions[i] for i in resdf['idx']]


if __name__ == '__main__':
    path_root = os.path.join(os.path.dirname(__file__), 'Dump')
    clear_folder(path_root)

    section_parameters = True
    if section_parameters:

        # New state:
        govnt_prty = dict()
        govnt_prty['A'] = 50
        govnt_prty['B'] = 20
        govnt_prty['X'] = 50

        govnt_disagree = dict()
        govnt_disagree['X'] = ['B']

        impossible_campaigns = list()
        impossible_moves = list()

    section_base_mode = True
    if section_base_mode:
        campaign_value = 2

        govt_state = State(campaign_value=campaign_value)
        govt_state.add_parties(govnt_prty)
        govt_state.add_restrictions(govnt_disagree)
        govt_state.run('max')
        max_shaps = govt_state.get_shapley()
        govt_state.to_csv(r'C:\school\PoliticalShapley\superadditive_or_max\Dump', title='MAX')

        govt_state = State(campaign_value=campaign_value)
        govt_state.add_parties(govnt_prty)
        govt_state.add_restrictions(govnt_disagree)
        govt_state.run('super_aditive')
        super_curr_shaps = govt_state.get_shapley()
        govt_state.to_csv(r'C:\school\PoliticalShapley\superadditive_or_max\Dump', title='SUPER')

        cdf = pd.DataFrame(super_curr_shaps, columns=['SuperAdditive'])
        cdf['max'] = max_shaps

        j = 3
