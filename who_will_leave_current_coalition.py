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
        # Current state:
        govnt_prty = dict()
        govnt_prty['Likud'] = 30
        govnt_prty['New Hope (Saar)'] = 6
        govnt_prty['Yamina (Bennett)'] = 7
        govnt_prty['Yesh Atid (Lapid)'] = 17
        govnt_prty['Joint list'] = 6
        govnt_prty['Shas'] = 9
        govnt_prty['United Torah Judaism'] = 7
        govnt_prty['Israel Beitenu'] = 7
        govnt_prty['Religious Zionists'] = 6
        govnt_prty['Meretz'] = 6
        govnt_prty['Avoda'] = 7
        govnt_prty['Blue and White (Gantz)'] = 8
        govnt_prty['Raam (Abbas)'] = 4

        govnt_disagree = dict()
        govnt_disagree['Likud'] = ['New Hope (Saar)', 'Blue and White (Gantz)', 'Israel Beitenu', 'Yesh Atid (Lapid)',
                                   'Avoda', 'Meretz']
        govnt_disagree['Joint list'] = ['Likud', 'Religious Zionists']
        govnt_disagree['Yesh Atid (Lapid)'] = ['Shas', 'United Torah Judaism']

        # New state:
        new_prty = dict()
        new_prty['Likud'] = 34
        new_prty['New Hope (Saar)'] = 0
        new_prty['Yamina (Bennett)'] = 6
        new_prty['Yesh Atid (Lapid)'] = 19
        new_prty['Joint list'] = 6
        new_prty['Shas'] = 9
        new_prty['United Torah Judaism'] = 7
        new_prty['Israel Beitenu'] = 6
        new_prty['Religious Zionists'] = 7
        new_prty['Meretz'] = 5
        new_prty['Avoda'] = 7
        new_prty['Blue and White (Gantz)'] = 9
        new_prty['Raam (Abbas)'] = 5

        new_disagree = dict()
        new_disagree['Likud'] = ['New Hope (Saar)', 'Blue and White (Gantz)', 'Yesh Atid (Lapid)', 'Avoda', 'Meretz',
                                 'Israel Beitenu']
        new_disagree['Joint list'] = ['Likud', 'Religious Zionists']
        new_disagree['Yesh Atid (Lapid)'] = ['Shas', 'United Torah Judaism']

        current_govt = ['New Hope (Saar)', 'Yamina (Bennett)', 'Yesh Atid (Lapid)', 'Israel Beitenu', 'Meretz', 'Avoda',
                        'Blue and White (Gantz)', 'Raam (Abbas)']
        current_opposition = [k for k in new_prty.keys() if k not in current_govt]

        impossible_campaigns = list()
        impossible_campaigns += [('Likud', 'Avoda'), ('Likud', 'Meretz'), ('Likud', 'Yesh Atid (Lapid)')]
        impossible_campaigns += [('Raam (Abbas)', 'Israel Beitenu'), ('Raam (Abbas)', 'Religious Zionists'),
                                 ('Raam (Abbas)', 'Yamina (Bennett)')]
        impossible_campaigns += [('Shas', 'Yesh Atid (Lapid)'), ('Shas', 'Israel Beitenu'), ('Shas', 'Meretz'),
                                 ('Shas', 'Avoda'), ('Shas', 'Blue and White (Gantz)')]
        impossible_campaigns += [('United Torah Judaism', 'Yesh Atid (Lapid)'),
                                 ('United Torah Judaism', 'Israel Beitenu'),
                                 ('United Torah Judaism', 'Meretz'),
                                 ('United Torah Judaism', 'Avoda'), ('United Torah Judaism', 'Blue and White (Gantz)')]

        impossible_moves = list()
        impossible_moves += [('Campaign', src, trgt) for (src, trgt) in impossible_campaigns]
        impossible_moves += [('Campaign', trgt, src) for (src, trgt) in impossible_campaigns]

    section_base_mode = True
    if section_base_mode:
        campaign_value = 2

        govt_state = State(campaign_value=campaign_value)
        govt_state.add_parties(govnt_prty)
        govt_state.add_restrictions(govnt_disagree)
        govt_state.run()
        curr_shaps = govt_state.get_shapley()

        new_state = State(campaign_value=campaign_value)
        new_state.add_parties(new_prty)
        new_state.add_restrictions(new_disagree)
        new_state.run()

        delta = StateDelta(old_state=govt_state, new_state=new_state, old_gov=current_govt)
        delta.zero_opposition_values()
        delta.shapdf.to_csv(os.path.join(path_root, 'Shap_old_vs_poll.csv'))
        delta.seatsdf.to_csv(os.path.join(path_root, 'seats_old_vs_poll.csv'))
        delta.new_state.legal_coalitions.to_csv(os.path.join(path_root, 'poll_legal_coalitions.csv'))

        player = 'Yamina (Bennett)'
        t_state = deepcopy(new_state)

    section_scan_all_moves = False
    if section_scan_all_moves:
        resdf = None
        significant_players = current_govt + ['Likud']
        for player in tqdm(significant_players):
            prev_score = t_state.get_score(player)
            res, actions = get_next_states(t_state, player)
            res['Actor'] = player
            print(tabulate(res, headers='keys', tablefmt='psql'))
            if resdf is None:
                resdf = res.copy()
            else:
                resdf = pd.concat([resdf, res])

            print("")
            time.sleep(0.1)

        resdf['in Govt'] = False
        resdf.loc[resdf['src'].isin(current_govt), 'in Govt'] = True
        resdf['src Govt gain'] = 0.0
        resdf['trgt Govt gain'] = 0.0
        for p_player in current_govt:
            resdf.loc[resdf['src'].eq(p_player), 'src Govt gain'] = curr_shaps.get(p_player, 0)
            resdf.loc[resdf['trgt'].eq(p_player), 'trgt Govt gain'] = curr_shaps.get(p_player, 0)

        resdf['Explode Govt'] = resdf['src Govt gain'] < resdf['src gain']
        resdf.to_csv(os.path.join(path_root, 'moves.csv'))

    section_apply_specific_moves = True
    if section_apply_specific_moves:
        resdf = None
        player = 'Likud'

        prev_score = t_state.get_score(player)
        res, possible_actions = get_next_states(t_state, player)

        # actions = [possible_actions[0]]
        actions = [
            ('Campaign', 'Yamina (Bennett)', 'Likud'),
            ('Campaign', 'Yamina (Bennett)', 'Likud'),
            ('Campaign', 'Raam (Abbas)', 'Yesh Atid (Lapid)'),
            ('Campaign', 'Raam (Abbas)', 'Joint list'),
        ]
        q_state = t_state.copy()
        for action in actions:
            print(f"Applying action: {action}")
            q_state = q_state.apply(action, campaign_power=1)

        delta = StateDelta(old_state=govt_state, new_state=q_state, old_gov=current_govt)
        delta.zero_opposition_values()
        delta.shapdf.to_csv(os.path.join(path_root, 'Shap_govnt_vs_actioned.csv'))
        delta.seatsdf.to_csv(os.path.join(path_root, 'seats_govnt_vs_actioned.csv'))
        delta.new_state.legal_coalitions.to_csv(os.path.join(path_root, 'actioned_legal_coalitions.csv'))
        res.to_csv(os.path.join(path_root, f'{player} moves.csv'))

        delta = StateDelta(old_state=new_state, new_state=q_state)
        delta.zero_opposition_values()
        delta.shapdf.to_csv(os.path.join(path_root, 'Shap_poll_vs_actioned.csv'))
        delta.seatsdf.to_csv(os.path.join(path_root, 'seats_poll_vs_actioned.csv'))

        print("")
        time.sleep(0.1)
