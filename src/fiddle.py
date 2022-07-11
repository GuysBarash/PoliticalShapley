import copy
import time

import numpy as np
import pandas as pd
from IPython.display import display
import json
import os
import shutil
import datetime
from tqdm import tqdm

import itertools
from copy import deepcopy
from tabulate import tabulate


def clear_folder(path, delete_if_exist=False):
    if delete_if_exist and os.path.exists(path):
        all_items_to_remove = [os.path.join(path, f) for f in os.listdir(path)]
        for item_to_remove in all_items_to_remove:
            if os.path.exists(item_to_remove) and not os.path.isdir(item_to_remove):
                os.remove(item_to_remove)
            else:
                shutil.rmtree(item_to_remove)

    if not os.path.exists(path):
        os.makedirs(path, mode=0o777)

    time.sleep(0.1)


def powerset(iterable):
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))


class PoliticalShapley:
    def __init__(self):
        self.coalitions = None
        self.coalitions_mandats = None
        self.coalitions_validity = None
        self.coalitions_value = None

        self.parties = dict()
        self.disagree = dict()

        self.shapley_values = None
        self.banzhf_values = None
        self.power_index = None
        self.legal_coalitions = None

        self.root_path = os.path.join(os.path.dirname(__file__), 'Dump')

    def add_parties(self, parties):
        self.parties.update(parties)
        self.parties = {k: v for k, v in self.parties.items() if v > 0}

    def add_restrictions(self, restrictions):
        self.disagree.update(restrictions)

    def _run_max(self):
        disagree = copy.deepcopy(self.disagree)
        for k in self.disagree.keys():
            vl = disagree[k]
            for v in vl:
                disagree[v] = list(set(disagree.get(v, list()) + [k]))

        binary_coalitions = np.array(list(map(list, itertools.product([0, 1], repeat=len(self.parties)))))
        self.coalitions = pd.DataFrame(index=range(binary_coalitions.shape[0]),
                                       columns=self.parties.keys(),
                                       data=binary_coalitions,
                                       )

        self.coalitions_mandats = self.coalitions.copy()
        self.coalitions_validity = self.coalitions.copy()
        self.coalitions_value = self.coalitions.copy()

        # Sum of each coalition
        for party, mandats in self.parties.items():
            self.coalitions_mandats[party] *= mandats
        self.coalitions_mandats['Total'] = self.coalitions_mandats.sum(axis=1)

        # Check validity
        self.coalitions_validity['invalid_count'] = 0
        for c_prty, antiprtys in self.disagree.items():
            if c_prty in self.coalitions_validity:
                prty_idx = self.coalitions_validity[c_prty].eq(1)
                cvdf = self.coalitions_validity[prty_idx]
                cvdf = cvdf[antiprtys]
                self.coalitions_validity.loc[cvdf.index, 'invalid_count'] += cvdf[antiprtys].sum(axis=1)

        self.coalitions_validity['valid'] = self.coalitions_validity['invalid_count'].eq(0)

        self.coalitions_value['value'] = 0
        good_coalitions = (self.coalitions_mandats['Total'] > 60)  # &  self.coalitions_validity['valid']
        self.coalitions_value.loc[good_coalitions, 'value'] = 100

        # Calculate shapley
        self.shapley_values = pd.Series(index=self.parties.keys())
        self.banzhf_values = pd.Series(index=self.parties.keys())
        for c_prty in self.parties.keys():
            curr_disagree = disagree.get(c_prty, list())
            other_parties = [cp for cp in self.parties.keys() if cp != c_prty]
            withdf = self.coalitions_value[self.coalitions_value[c_prty].eq(1)]
            withoutdf = self.coalitions_value[self.coalitions_value[c_prty].eq(0)]
            reordered = pd.concat([withdf, withoutdf])
            reordered = reordered.sort_values(by=other_parties)
            reordered['value_shift'] = reordered['value'].shift(-1).fillna(0)

            reordered['gain'] = (reordered['value'] - reordered['value_shift']).fillna(0)
            if len(curr_disagree) > 0:
                reordered.loc[reordered[curr_disagree].sum(axis=1) > 0, 'gain'] = 0

            reordered = reordered.loc[self.coalitions_value[c_prty].eq(1), [c for c in reordered.columns
                                                                            if c != 'qvalue']]

            if c_prty == 'A':
                t = reordered[reordered['gain'] > 0]
                td = reordered[(reordered['value_shift'] < 50) & (reordered['value'] > 50)]
                print(f"[MAX RUN][{c_prty}]\t Sum gain {reordered['gain'].sum()}")
            n = len(self.parties)
            bnzf_coef = 1.0 / (np.power(2, n - 1))  # 1/(2^(n-1))
            reordered['N'] = len(self.parties)
            reordered['S'] = reordered[other_parties].sum(axis=1)

            n_fac = np.math.factorial(len(self.parties))
            s_fac = reordered['S'].apply(np.math.factorial)
            comp_s_fac = (reordered['N'] - reordered['S'] - 1).apply(np.math.factorial)
            gains = reordered['gain']

            shap_gain = ((s_fac * comp_s_fac).astype(float) / float(n_fac)) * gains
            shap_val = shap_gain.sum()

            banzhf_gain = gains.sum()
            banzhf_val = bnzf_coef * banzhf_gain

            self.shapley_values[c_prty] = shap_val
            self.banzhf_values[c_prty] = banzhf_val

        self.shapley_values = self.shapley_values.sort_values(ascending=False).fillna(0)
        self.banzhf_values = self.banzhf_values.sort_values(ascending=False).fillna(0)
        self.power_index = pd.DataFrame(columns=['Shapley', 'Banzhf'], index=self.parties)
        self.power_index['Shapley'] = self.shapley_values
        self.power_index['Banzhf'] = self.banzhf_values
        self.power_index = self.power_index.sort_values(by=['Shapley', 'Banzhf'], ascending=False)
        self.legal_coalitions = self.coalitions_mandats.loc[self.coalitions_value['value'] > 0].reset_index(drop=True)

    def _run_super_additive(self):
        binary_coalitions = np.array(list(map(list, itertools.product([0, 1], repeat=len(self.parties)))))
        self.coalitions = pd.DataFrame(index=range(binary_coalitions.shape[0]),
                                       columns=self.parties.keys(),
                                       data=binary_coalitions,
                                       )

        self.coalitions_mandats = self.coalitions.copy()
        self.coalitions_validity = self.coalitions.copy()
        self.coalitions_value = self.coalitions.copy()

        # Sum of each coalition
        for party, mandats in self.parties.items():
            self.coalitions_mandats[party] *= mandats
        self.coalitions_mandats['Total'] = self.coalitions_mandats.sum(axis=1)

        # Check validity
        self.coalitions_validity['invalid_count'] = 0
        for c_prty, antiprtys in self.disagree.items():
            if c_prty in self.coalitions_validity:
                prty_idx = self.coalitions_validity[c_prty].eq(1)
                cvdf = self.coalitions_validity[prty_idx]
                antiprtys = [ap for ap in antiprtys if ap in cvdf.columns]
                cvdf = cvdf[antiprtys]
                self.coalitions_validity.loc[cvdf.index, 'invalid_count'] += cvdf[antiprtys].sum(axis=1)

        self.coalitions_validity['valid'] = self.coalitions_validity['invalid_count'].eq(0)

        self.coalitions_value['value'] = 0
        good_coalitions = self.coalitions_validity['valid'] & (self.coalitions_mandats['Total'] > 60)
        good_coalitionsdf = self.coalitions_validity[good_coalitions]
        self.coalitions_value.loc[good_coalitions, 'value'] = 100

        # Memorize all coalitions value using super additivity
        section_memorize = True
        if section_memorize:
            # That means that the value of each coalition is it's best value of any subcoalition
            sig_sr = pd.DataFrame(index=self.coalitions_value.index)
            sig_sr['seats'] = self.coalitions_mandats['Total']
            sig_sr['value'] = 0
            sig_sr.loc[good_coalitions, 'value'] = 1
            sig_sr['supervalue'] = -1
            sig_sr.loc[sig_sr['seats'] < 61, 'supervalue'] = 0

            gooddf = sig_sr.loc[good_coalitions]
            for goodidx in tqdm(gooddf.index.to_list(), desc='Supergroups', disable=True):
                gsr = self.coalitions.loc[goodidx]
                good_cols = gsr[gsr > 0].index.to_list()
                c_size = len(good_cols)

                supergroups = self.coalitions[good_cols].sum(axis=1) == c_size
                sig_sr.loc[supergroups, 'supervalue'] = 1
            sig_sr.loc[sig_sr['supervalue'] == -1, 'supervalue'] = 0
            good_coalitions = sig_sr['supervalue'] > 0
            self.coalitions_value.loc[good_coalitions, 'value'] = 100

        # Calculate shapley
        self.shapley_values = pd.Series(index=self.parties.keys(), dtype=float)
        self.banzhf_values = pd.Series(index=self.parties.keys(), dtype=float)
        for c_prty in self.parties.keys():
            other_parties = [cp for cp in self.parties.keys() if cp != c_prty]
            withdf = self.coalitions_value[self.coalitions_value[c_prty].eq(1)]
            withoutdf = self.coalitions_value[self.coalitions_value[c_prty].eq(0)]
            reordered = pd.concat([withdf, withoutdf])
            reordered = reordered.sort_values(by=other_parties)
            reordered['value_shift'] = reordered['value'].shift(-1)
            reordered['gain'] = (reordered['value'] - reordered['value_shift'])
            reordered = reordered.loc[self.coalitions_value[c_prty].eq(1), [c for c in reordered.columns
                                                                            if c != 'qvalue']]

            if c_prty == 'A':
                t = reordered[reordered['gain'] > 0]
                td = reordered[(reordered['value_shift'] < 50) & (reordered['value'] > 50)]
                print(f"[SUPER RUN][{c_prty}]\t Sum gain {reordered['gain'].sum()}")

            n = len(self.parties)
            bnzf_coef = 1.0 / (np.power(2, n - 1))  # 1/(2^(n-1))
            reordered['N'] = len(self.parties)
            reordered['S'] = reordered[other_parties].sum(axis=1)

            n_fac = np.math.factorial(len(self.parties))
            s_fac = reordered['S'].apply(np.math.factorial)
            comp_s_fac = (reordered['N'] - reordered['S'] - 1).apply(np.math.factorial)
            gains = reordered['gain']

            shap_gain = ((s_fac * comp_s_fac).astype(float) / float(n_fac)) * gains
            shap_val = shap_gain.sum()

            banzhf_gain = gains.sum()
            banzhf_val = bnzf_coef * banzhf_gain

            self.shapley_values[c_prty] = shap_val
            self.banzhf_values[c_prty] = banzhf_val

        self.shapley_values = self.shapley_values.sort_values(ascending=False).fillna(0)
        self.banzhf_values = self.banzhf_values.sort_values(ascending=False).fillna(0)
        self.power_index = pd.DataFrame(columns=['Shapley', 'Banzhf'], index=self.parties)
        self.power_index['Shapley'] = self.shapley_values
        self.power_index['Banzhf'] = self.banzhf_values
        self.power_index = self.power_index.sort_values(by=['Shapley', 'Banzhf'], ascending=False)
        self.legal_coalitions = self.coalitions_mandats.loc[
            self.coalitions_validity['valid'] & (self.coalitions_mandats['Total'] > 60)].reset_index(drop=True)

    def run(self, sum_function='Super Additive'):
        if sum_function == 'max':
            self._run_max()
        else:
            self._run_super_additive()

    def get_mandates(self, prty=None):
        if prty is not None:
            return self.parties.get(prty, None)
        else:
            df = pd.DataFrame(pd.Series(self.parties, name='Mandates'))
            df = df.sort_values(by='Mandates', ascending=False)
            return df

    def get_possible_govt(self):
        return self.legal_coalitions

    def get_shapley(self, party=None):
        if party is None:
            return self.shapley_values
        else:
            return self.shapley_values.get(party, None)

    def to_csv(self, path=None, title=None):
        if path is None:
            path = self.root_path
        if title is None:
            title = ''
        else:
            title = f'{title} '

        shap_path = os.path.join(path, f'{title}Power index.csv')
        self.power_index.to_csv(shap_path, header=True)

        possible_govt_path = os.path.join(path, f'{title}possible options.csv')
        self.legal_coalitions.to_csv(possible_govt_path)


def get_campagin_tactics(base, prty):
    prts = [k for k in base.parties.keys() if k != prty]
    restrictions = base.govnt_disagree

    mx_mandates_to_steal = 4
    sr = pd.DataFrame(columns=range(1, mx_mandates_to_steal + 1), index=prts)
    for foe in tqdm(prts, desc='Calculating attack tactics'):
        mx_mandates_to_steal_t = min(mx_mandates_to_steal, base.parties[foe])
        for mndts_stolen in range(1, mx_mandates_to_steal_t + 1):
            new_scores = base.parties.copy()
            new_scores[foe] -= mndts_stolen
            new_scores[prty] += mndts_stolen
            new_scores_sr = pd.Series(new_scores)

            shap_t = PoliticalShapley()
            shap_t.add_parties(new_scores)
            shap_t.add_restrictions(restrictions)
            shap_t.run()
            shaps = shap_t.get_shapley()
            coalitions = shap_t.get_possible_govt()
            coalitions_with_prty = coalitions[coalitions[prty] > 0]
            coalitions_without_prty = coalitions[coalitions[prty] <= 0]
            sr.loc[foe, mndts_stolen] = shap_t.get_shapley(prty)

    for i in range(1, mx_mandates_to_steal + 1):
        if sr[i].unique().shape[0] > 1:
            sr = sr.sort_values(by=list(range(1, mx_mandates_to_steal + 1)[i - 1:]), ascending=False)
            break
    return sr


def get_weak_spots(base, prty):
    prts = [k for k in base.parties.keys() if k != prty]
    restrictions = base.govnt_disagree.copy()

    mx_mandates_to_steal = 4
    mx_mandates_to_steal = min(mx_mandates_to_steal, base.parties[prty] - 1)
    sr = pd.DataFrame(columns=range(1, mx_mandates_to_steal + 1), index=prts)
    for foe in tqdm(prts, desc='Exploring weak spots'):
        mx_mandates_to_steal_t = min(mx_mandates_to_steal, base.parties[prty] - 1)
        for mndts_stolen in range(1, mx_mandates_to_steal_t + 1):
            new_scores = base.parties.copy()
            new_scores[foe] += mndts_stolen
            new_scores[prty] -= mndts_stolen
            new_scores_sr = pd.Series(new_scores)

            shap_t = PoliticalShapley()
            shap_t.add_parties(new_scores)
            shap_t.add_restrictions(restrictions)
            shap_t.run()
            shaps = shap_t.get_shapley()
            coalitions = shap_t.get_possible_govt()
            coalitions_with_prty = coalitions[coalitions[prty] > 0]
            coalitions_without_prty = coalitions[coalitions[prty] <= 0]
            sr.loc[foe, mndts_stolen] = shap_t.get_shapley(prty)

            # if foe == 'New Hope (Saar)':
            #     shap_t.to_csv(title='Likud_attacked')

    for i in range(1, mx_mandates_to_steal + 1):
        if sr[i].unique().shape[0] > 1:
            sr = sr.sort_values(by=list(range(1, mx_mandates_to_steal + 1)[i - 1:]), ascending=True)
            break
    return sr


def which_rule_to_break(base, prty):
    scores = base.parties.copy()
    disagree = list()

    for prty_t in base.govnt_disagree.keys():
        if prty_t == prty:
            disagree += base.govnt_disagree[prty_t]
        else:
            if prty in base.govnt_disagree[prty_t]:
                disagree += [prty_t]

    df = pd.DataFrame(index=disagree, columns=['Old Shap', 'New Shap', 'Shap gain', 'options with', 'options without'])
    for disagree_prty in tqdm(disagree, desc='Who is likely to join Likud'):
        new_disagree = base.govnt_disagree.copy()
        if disagree_prty in new_disagree.get(prty, list()):
            new_l = [t for t in new_disagree[prty] if t != disagree_prty]
            new_disagree[prty] = new_l
        if prty in new_disagree.get(disagree_prty, list()):
            new_l = [t for t in new_disagree[disagree_prty] if t != prty]
            new_disagree[disagree_prty] = new_l

        shap_t = PoliticalShapley()
        shap_t.add_parties(scores)
        shap_t.add_restrictions(new_disagree)
        shap_t.run()
        shaps = shap_t.get_shapley()
        coalitions = shap_t.get_possible_govt()
        coalitions_with_prty = coalitions[coalitions[prty] > 0]
        coalitions_without_prty = coalitions[coalitions[prty] <= 0]

        df.loc[disagree_prty, 'Old Shap'] = base.get_shapley(disagree_prty)
        df.loc[disagree_prty, 'New Shap'] = shap_t.get_shapley(disagree_prty)
        df.loc[disagree_prty, 'Shap gain'] = shap_t.get_shapley(disagree_prty) / base.get_shapley(disagree_prty)
        df.loc[disagree_prty, ['options with', 'options without']] = len(coalitions_with_prty), len(
            coalitions_without_prty)
        if disagree_prty in ['New Hope (Saar)', 'Meretz']:
            shap_t.to_csv(title=f'{disagree_prty}_joined_Likud')

    df = df.sort_values(by='Shap gain', ascending=False)
    return df


def who_has_interest_to_attack(base, prty):
    prts = [k for k in base.parties.keys() if k != prty]
    restrictions = base.govnt_disagree.copy()
    scores = base.parties.copy()
    original_shaps = base.get_shapley()

    mx_mandates_to_steal = 4
    sr = pd.DataFrame(columns=range(1, mx_mandates_to_steal + 1), index=prts)
    for foe in tqdm(prts, desc='Calculating agendas'):
        mx_mandates_to_steal_t = min(mx_mandates_to_steal, base.parties[foe])
        for mndts_stolen in range(1, mx_mandates_to_steal_t + 1):
            new_scores = base.parties.copy()
            new_scores[foe] += mndts_stolen
            new_scores[prty] -= mndts_stolen
            new_scores_sr = pd.Series(new_scores)

            shap_t = PoliticalShapley()
            shap_t.add_parties(new_scores)
            shap_t.add_restrictions(restrictions)
            shap_t.run()
            shaps = shap_t.get_shapley()
            foe_prev_shap = original_shaps[foe]
            foe_new_shap = shaps[foe]
            if foe_prev_shap == 0:
                attack_gain = 1.0
                if foe_new_shap > 0:
                    attack_gain = 2.0
            else:
                attack_gain = foe_new_shap / foe_prev_shap
            sr.loc[foe, mndts_stolen] = attack_gain

    for i in range(1, mx_mandates_to_steal + 1):
        if sr[i].unique().shape[0] > 1:
            sr = sr.sort_values(by=list(range(1, mx_mandates_to_steal + 1)[i - 1:]), ascending=False)
            break
    return sr


def dict_to_latex_table(parties, dispute, print_parties=False, print_restrictions=False, exit_on_complete=False):
    ld = list(parties.items())
    ld.sort(key=lambda t: -t[1])
    ld = [(i + 1, t[0].replace('_', ' ').title(), t[1]) for i, t in enumerate(ld)]
    parties_idx = {t[1]: t[0] for t in ld}

    if print_parties:
        ld = list(parties.items())
        ld.sort(key=lambda t: -t[1])
        ld = [(t[0].replace('_', ' ').title(), t[1]) for t in ld]
        ld_s = [f'${idx}$ & {t[0]} & ${t[1]}$' for idx, t in enumerate(ld)]
        s = ' \\\\\n'.join(ld_s)

        q = [str(t[1]) for t in ld]
        qs = ','.join(q)
        print(f"(61;{qs})")
        print(s)
    if print_restrictions:
        l = list()
        for k, v in dispute.items():
            k_fixed = k.replace('_', ' ').title()
            for vt in v:
                vt_fixed = vt.replace('_', ' ').title()
                t = (k_fixed, vt_fixed)
                t_r = (vt_fixed, k_fixed)
                if (t not in l) and (t_r not in l):
                    l += [t]

        l_s = [f' {t[0]} & {parties_idx[t[0]]} & {t[1]} & {parties_idx[t[1]]}' for idx, t in enumerate(l)]
        s = ' \\\\\n'.join(l_s)
        print(s)

        l_s = [f'({parties_idx[t[0]]}, {parties_idx[t[1]]})' for t in l]
        s = f'({", ".join(l_s)})'
        print(s)

    if exit_on_complete:
        exit(0)


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
        self.voter = None
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

        return list(set(self.disagree.keys()))

    def get_score(self, player=None):
        if player is not None:
            shaps = self.get_shapley(player)
            mandates = self.get_mandates(player)
        else:
            shaps = self.get_shapley()
            mandates = self.get_mandates()
        return shaps

    def get_actions_poiter_at(self, player):
        if player not in self.get_players():
            # raise Exception(f"No such player: {player}")
            return list()

        campaign_actions = list()
        for src in self.get_players():
            if src == player:
                continue
            else:
                if self.parties.get(src, 0) > 0:
                    t_action = ('Campaign', src, player)
                    campaign_actions.append(t_action)

        if self.impossible_moves is not None:
            campaign_actions = [t for t in campaign_actions if t not in self.impossible_moves]
        elif self.voter is not None:
            pass
        else:
            pass

        breaking_promise_actions = list()
        for p1 in self.disagree.keys():
            for p2 in self.disagree[p1]:
                if p1 == player or p2 == player:
                    continue
                else:
                    t_action = ('Break', p1, p2)
                    breaking_promise_actions.append(t_action)

        ret = breaking_promise_actions + campaign_actions
        return list(set(ret))

    def get_actions(self, player):
        if player not in self.get_players():
            # raise Exception(f"No such player: {player}")
            return list()

        campaign_actions = list()
        for trgt in self.get_players():
            if trgt == player:
                continue
            else:
                if self.parties.get(trgt, 0) > 0:
                    t_action = ('Campaign', player, trgt)
                    campaign_actions.append(t_action)

        if self.impossible_moves is not None:
            campaign_actions = [t for t in campaign_actions if t not in self.impossible_moves]
        elif self.voter is not None:
            pass
            # res = list()
            # for action_type, action_src, action_trgt in campaign_actions:
            #     action_value, action_details = self.voter.evaluate_campaign(action_src, action_trgt)
            #     res.append([action_src, action_trgt, action_value, action_details])
            # res.sort(key=lambda t: -t[2])
            # for action_src, action_trgt, action_value, action_details in res:
            #     print(f'{action_src} --> {action_trgt}\t{action_value:>.2f}')

        else:
            pass

        restrictions = self.get_restrictions(player)
        breaking_promise_actions = list()
        for adv in restrictions:
            t_action = ('Break', player, adv)
            breaking_promise_actions.append(t_action)

        ret = breaking_promise_actions + campaign_actions
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

    def add_voter(self, voter=None):
        if voter is None:
            self.voter = Voter()
            self.voter.stats_import(translate=True)
        else:
            self.voter = voter

    def evaluate_campaign_by_voters(self, src, trgt):
        ret = self.voter.evaluate_campaign(src, trgt)
        return ret


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


class Voter:
    def __init__(self):
        self.rootpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'statistics')
        self.outpath = os.path.join(self.rootpath, 'Results')
        self.datapath = os.path.join(self.rootpath, 'data')
        self.lms_path = os.path.join(self.datapath, 'lms')
        clear_folder(self.outpath, delete_if_exist=False)
        clear_folder(self.rootpath, delete_if_exist=False)
        clear_folder(self.lms_path, delete_if_exist=False)

        self.rawdf = None
        self.ballots = None
        self.metadata = None
        self.parties = None
        self.voters = None
        self.town_clusters = None
        self.votesdf = None
        self.votes_dist_df = None
        self.non_empy_parties = None
        self.parties_names_translation = None
        self.distances = None
        self.k = None

    def initialize(self, votes_per_settelments_path, ballots_path=None):
        df = pd.read_csv(votes_per_settelments_path, encoding='iso_8859_8')
        self.rawdf = df.copy()
        self.rawdf = self.rawdf[~self.rawdf['שם ישוב'].eq("מעטפות חיצוניות")]
        self.rawdf.loc[:, 'uid'] = self.rawdf.loc[:, 'סמל ישוב'].astype(int).values

        self.ballots = None
        if ballots_path is not None:
            with open(ballots_path, encoding='utf-8') as json_file:
                self.ballots = json.load(json_file)
            self.rawdf = self.rawdf.rename(columns=self.ballots)

        self.metadata = list(set(df.columns[:7].to_list() + ['uid']))
        self.parties = [c for c in self.rawdf.columns if c not in self.metadata]
        self.parties = [c for c in self.parties if 'Unnamed' not in c]

        self.votesdf = self.rawdf[self.parties]
        self.votesdf = self.votesdf[self.votesdf.sum(axis=1) > 500]
        # self.votesdf = self.votesdf[self.votesdf.sum(axis=1) < 4000]
        names = self.rawdf.loc[self.votesdf.index, 'שם ישוב']
        uid = self.rawdf.loc[self.votesdf.index, 'uid']

        self.votes_dist_df = self.votesdf.div(self.votesdf.sum(axis=1), axis=0)
        self.votes_dist_df = np.round(self.votes_dist_df, 2)
        self.votes_dist_df = self.votes_dist_df.div(self.votes_dist_df.sum(axis=1), axis=0)

        nonempties = self.votes_dist_df.max() > 0
        nonempties = nonempties[nonempties]
        self.non_empy_parties = nonempties
        self.votes_dist_df = self.votes_dist_df[nonempties.index]
        self.votes_dist_df['town'] = names
        self.votes_dist_df['uid'] = uid
        self.votes_dist_df = self.votes_dist_df.set_index('town')

        d = self.votes_dist_df[[c for c in self.votes_dist_df if c in self.parties]].to_numpy()
        dist = euclidean_distances(d, d)
        self.distances = pd.DataFrame(index=names, columns=names, data=dist)

    def add_external_data(self):
        cols_to_mean = list()
        cols_to_most_common = list()

        section_ethnicity = False
        if section_ethnicity:
            ethdf = pd.read_csv(os.path.join(self.lms_path, 'ethnicity.csv'), encoding='utf-8-sig')
            ethdf['uid'] = ethdf['סמל היישוב'].str.replace(',', '').astype(int)

            edf = pd.DataFrame(index=ethdf.index)
            edf['uid'] = ethdf['uid']
            edf['Jews'] = ethdf['יהודים ואחרים (אחוזים)'].str.replace('-', '0.0').astype(float) / 100.0
            edf['Arab'] = ethdf['ערבים (אחוזים)'].str.replace('-', '0.0').replace('', '0.0').astype(float) / 100.0
            edf['Arab-Muslim'] = ethdf['מוסלמים (אחוזים מתוך האוכלוסייה הערבית)'].str.replace('-', '0.0').replace('',
                                                                                                                  '0.0').astype(
                float) / 100.0
            edf['Arab-Muslim'] *= edf['Arab']
            edf['Arab-Christian'] = ethdf['נוצרים (אחוזים מתוך האוכלוסייה הערבית)'].str.replace('-', '0.0').replace('',
                                                                                                                    '0.0').astype(
                float) / 100.0
            edf['Arab-Christian'] *= edf['Arab']
            edf['Druze'] = ethdf['דרוזים (אחוזים מתוך האוכלוסייה הערבית)'].str.replace('-', '0.0').replace('',
                                                                                                           '0.0').astype(
                float) / 100.0
            edf['Druze'] *= edf['Arab']
            edf = edf.drop('Arab', axis=1)

            self.votesdf = self.votesdf.merge(edf, left_on='uid', right_on='uid', how='outer')

        section_money = True
        if section_money:
            money = pd.read_csv(os.path.join(self.lms_path, 'paycheck.csv'), encoding='utf-8-sig')
            money['uid'] = money['סמל היישוב'].str.replace(',', '').astype(int)
            money['Salary'] = money['שכר ממוצע לחודש של שכירים (ש"ח) כלל השכירים'].str.replace(',', '').astype(float)
            money['JINI'] = money["מדד אי-השוויון שכירים (מדד ג'יני, 0 שוויון מלא)"].astype(float)
            money = money[['uid', 'Salary', 'JINI']]
            self.votesdf = self.votesdf.merge(money, left_on='uid', right_on='uid', how='outer')

        section_education = True
        if section_education:
            education = pd.read_csv(os.path.join(self.lms_path, 'education.csv'), encoding='utf-8-sig')
            education = education[~education['סמל היישוב'].isna()]
            education['uid'] = education['סמל היישוב'].str.replace(',', '').astype(int)
            education['Bagrut'] = education['אחוז זכאים לתעודת בגרות מבין תלמידי כיתות יב תשע"ט 2018/19'].str.replace(
                '-', '0.0').astype(float)
            education['Degree'] = education['השכלה גבוהה אחוז סטודנטים מתוך אוכלוסיית בני 25-20 תש"ף 2019/20']
            education = education[['uid', 'Bagrut', 'Degree']]
            self.votesdf = self.votesdf.merge(education, left_on='uid', right_on='uid', how='outer')

        section_health = True
        if section_health:
            health = pd.read_csv(os.path.join(self.lms_path, 'health.csv'), encoding='utf-8-sig')
            health = health[~health['סמל היישוב'].isna()]

            q = pd.DataFrame(index=health.index)
            q['uid'] = health['סמל היישוב'].str.replace(',', '').astype(int)
            q['under 18'] = health['teens'] / 100.0
            q['above 75'] = health['olds'] / 100.0
            q['natural growth'] = health['growth normal']
            q['density'] = health['Density'].str.replace(',', '').apply(pd.to_numeric, args=('coerce',))
            self.votesdf = self.votesdf.merge(q, left_on='uid', right_on='uid', how='outer')

        section_age = True
        if section_age:
            age = pd.read_csv(os.path.join(self.lms_path, 'age.csv'), encoding='utf-8-sig')
            age = age[~age['uid'].isna()]

            age_bins = [c for c in age.columns if c not in ['town', 'uid', 'total population']]
            q = age[[c for c in age.columns if c not in ['town']]]
            for c in ['age 0 to 5', 'age 5 to 18', 'age 19 to 45', 'age 46 to 55', 'age 56 to 64', 'age 65 and above']:
                q[c] = q[c].astype(float) / q['total population'].astype(float)
            self.votesdf = self.votesdf.merge(q, left_on='uid', right_on='uid', how='outer')
            cols_to_mean += age_bins + ['total population']

        section_general = True
        if section_general:
            gendf = pd.read_csv(os.path.join(self.lms_path, 'general.csv'), encoding='utf-8-sig')
            gendf = gendf[~gendf['סמל היישוב'].isna()]

            q = pd.DataFrame(index=gendf.index)
            q['uid'] = gendf['סמל היישוב']
            q['county'] = gendf['נפה']
            q['Settelment type'] = gendf['צורת יישוב']
            q['population'] = gendf['אוכלוסייה - סך הכל'].str.replace(',', '').str.replace('-', '0').astype(float)
            q['Jews pop'] = gendf['מזה: יהודים ואחרים'].str.replace(',', '').str.replace('-', '0').astype(float) / q[
                'population']
            q['Non-Jews pop'] = gendf['מזה: ערבים'].str.replace(',', '').str.replace('-', '0').astype(float) / q[
                'population']
            self.votesdf = self.votesdf.merge(q, left_on='uid', right_on='uid', how='outer')

        section_crime = True
        if section_crime:
            crime = pd.read_csv(os.path.join(self.lms_path, 'police_info.csv'), encoding='utf-8-sig')
            for t in crime.columns:
                if 'Unnamed' in t:
                    crime = crime.drop(t, axis=1)

            cols = ['Crime public security', 'Crime moral', 'Crime property',
                    'Crime sex', 'Crime cheat', 'Crime people', 'Crime body',
                    'Crime public order', 'Crime license', 'Crime financial',
                    'Crime driving', 'Crime beuracracy', 'Crime of definition',
                    'Crime other']
            q = pd.DataFrame(index=crime.index)
            q['uid'] = crime['uid'].astype(int)
            q['Crime public security'] = crime['עבירות בטחון'].astype(int)
            q['Crime moral'] = crime['עבירות כלפי המוסר'].astype(int)
            q['Crime property'] = crime['עבירות כלפי הרכוש'].astype(int)
            q['Crime sex'] = crime['עבירות מין'].astype(int)
            q['Crime cheat'] = crime['עבירות מרמה'].astype(int)
            q['Crime people'] = crime['עבירות נגד אדם'].astype(int)
            q['Crime body'] = crime['עבירות נגד גוף'].astype(int)
            q['Crime public order'] = crime['עבירות סדר ציבורי'].astype(int)
            q['Crime license'] = crime['עבירות רשוי'].astype(int)
            q['Crime financial'] = crime['עבירות כלכליות'].astype(int)
            q['Crime driving'] = crime['עבירות תנועה'].astype(int)
            q['Crime beuracracy'] = crime['עבירות מנהליות'].astype(int)
            q['Crime of definition'] = crime['סעיפי הגדרה'].astype(int)
            q['Crime other'] = crime['שאר עבירות'].astype(int)
            q['Crimes total'] = q[cols].sum(axis=1)
            self.votesdf = self.votesdf.merge(q, left_on='uid', right_on='uid', how='outer')

            cols_to_normal = cols + ['Crimes total']
            cols_to_mean += cols + ['Crimes total']
            for c in cols_to_normal:
                self.votesdf[c] = (100 * self.votesdf[c].astype(float)) / self.votesdf['population']

        section_mosques = True
        if section_mosques:
            reldf = pd.read_csv(os.path.join(self.lms_path, 'Mosques.csv'), encoding='utf-8-sig', index_col=0)

            self.votesdf = self.votesdf.merge(reldf, left_on='uid', right_on='uid', how='outer')
            cols_to_mean += ['Mosques']
            cols_to_normal = ['Mosques']
            for c in cols_to_normal:
                self.votesdf[c] = (100 * self.votesdf[c].astype(float)) / self.votesdf['population'].astype(float)

        self.votesdf = self.votesdf[~self.votesdf['Label'].isna()]
        self.voters['Leading party'] = self.voters[[c for c in self.voters.columns if c in self.parties]].idxmax(
            axis=1)

        section_add_means = True
        if section_add_means:
            cols_to_mean += ['Jews', 'Arab-Muslim', 'Arab-Christian', 'Druze',
                             'Bagrut', 'Degree',
                             'under 18', 'above 75', 'natural growth', 'density',
                             'Salary', 'JINI',
                             'population', 'Jews pop', 'Non-Jews pop',
                             ]
            for c in cols_to_mean:
                if c not in self.votesdf.columns:
                    continue
                df = self.votesdf[['Label', c]]
                g = df.groupby('Label')
                nullratio = g.agg({c: lambda x: x.isnull().mean()})
                nullratio = nullratio[nullratio < 0.6].dropna()
                self.voters.loc[nullratio.index.astype(int).to_list(), c] = g[c].median()

        section_add_most_common = True
        if section_add_most_common:
            cols_to_most_common += ['county', 'Settelment type'
                                    ]
            for c in cols_to_most_common:
                if c not in self.votesdf.columns:
                    continue
                df = self.votesdf[['Label', c]]
                g = df.groupby('Label')

                nullratio = g.agg({c: lambda x: x.isnull().mean()})
                nullratio = nullratio[nullratio < 0.5].dropna()
                self.voters.loc[nullratio.index.astype(int).to_list(), c] = g[c].agg(
                    lambda x: x.value_counts().index[0])

        j = 3

    def cluster(self):
        # dbscan clustering
        from kneed import KneeLocator
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        prts = [c for c in self.votes_dist_df if c in self.parties]
        X = self.votes_dist_df[prts].to_numpy()

        scaler = StandardScaler()
        scaled_X = X  # scaler.fit_transform(X)
        opt_k = 9

        section_calculate_optimal_k = False
        if section_calculate_optimal_k:
            k_range = (3, 30)
            sse = pd.Series(index=range(k_range[0], k_range[1]))
            silhouette = pd.Series(index=range(k_range[0], k_range[1]))

            for k in tqdm(range(k_range[0], k_range[1]), desc='Calculating optimal K'):
                model = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=42)
                model.fit(scaled_X)
                sse[k] = model.inertia_
                silhouette[k] = silhouette_score(scaled_X, model.labels_)

            kl = KneeLocator(range(k_range[0], k_range[1]), sse, curve="convex", direction="decreasing")
            opt_k = kl.elbow + 1

            fig, axs = plt.subplots(2)
            fig.suptitle('Vertically stacked subplots')
            axs[0].plot(range(k_range[0], k_range[1]), sse)
            _ = axs[0].vlines(x=opt_k, ymin=sse.min() * 0.5, ymax=sse.max() * 1.2, colors='r')
            axs[0].set_title('SSE')

            axs[1].plot(range(k_range[0], k_range[1]), silhouette)
            axs[1].set_title('silhouette')
            _ = axs[1].vlines(x=opt_k, ymin=silhouette.min() * 0.5, ymax=silhouette.max() * 1.2, colors='r')

            plt.style.use("fivethirtyeight")
            plt.xticks(range(k_range[0], k_range[1]))
            # plt.show()
            figpath = os.path.join(self.outpath, 'optimal K for clustering')
            plt.savefig(figpath)

        print(f'Types of voters: {opt_k}')
        self.k = opt_k
        model = KMeans(init="random", n_clusters=opt_k, n_init=10, max_iter=300, random_state=42)
        model.fit(scaled_X)
        self.votes_dist_df['Label'] = model.labels_
        self.votes_dist_df['Votes'] = self.votesdf.sum(axis=1).values
        self.votesdf['Label'] = self.votes_dist_df['Label'].values

        self.votes_dist_df = self.votes_dist_df.sort_values(by='Label', ascending=False)
        voters = self.votes_dist_df[prts + ['Label']].groupby('Label').mean()
        voters['Votes'] = self.votes_dist_df.groupby('Label')['Votes'].sum()
        voters['Votes ratio'] = voters['Votes'] / voters['Votes'].sum()

        cols_to_norm = list(self.non_empy_parties.index) + ['Votes ratio']
        # voters[cols_to_norm] *= 100
        # voters[cols_to_norm] = np.round(voters[cols_to_norm], 1)
        voters[cols_to_norm] = np.round(voters[cols_to_norm], 3)

        print('<-------------------------->')
        for cls in range(opt_k):
            xdf = self.votes_dist_df[self.votes_dist_df['Label'] == cls]
            t = xdf.iloc[:6].index.to_list()
            print(f'--- {cls} ---')
            for tq in t:
                print(tq)
        print('<-------------------------->')

        nonempties = voters.max() > 0
        nonempties = nonempties[nonempties]
        voters = voters[nonempties.index]

        for c in self.metadata:
            self.votesdf[c] = 0
        self.votesdf.loc[:, self.metadata] = self.rawdf.loc[self.votesdf.index, self.metadata]

        self.voters = voters
        self.voters['Towns count'] = self.votesdf.groupby('Label')['uid'].count()
        self.add_external_data()

    def export(self):
        for df, title in [
            (self.votes_dist_df, 'votes distributions'),
            (self.voters, 'Meta voters'),
            (self.votesdf, 'votes unnormalized'),
            (self.rawdf, 'Raw data'),
            (self.distances, 'town distances'),

        ]:
            if df is not None:
                df.to_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        # Towns by label
        df = self.votesdf[['Label', 'שם ישוב']]
        tdf = pd.DataFrame(columns=sorted(df['Label'].unique()), index=range(df['Label'].value_counts().max()))
        for c in df['Label'].unique():
            r = df[df['Label'].eq(c)]['שם ישוב'].values
            tdf.loc[range(len(r)), c] = df[df['Label'].eq(c)]['שם ישוב'].values

        self.town_clusters = tdf
        title = 'Towns cluster'
        tdf.to_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'notes signs'
        with open(os.path.join(self.outpath, f'{title}.json'), 'w', encoding='utf-8') as file:
            json.dump(self.ballots, file, ensure_ascii=False, indent=4)

        # Additional information
        info = dict()
        info['metadata'] = self.metadata
        info['parties'] = self.parties
        info['non_empy_parties'] = self.non_empy_parties.index.to_list()
        info['k'] = self.k
        title = 'additional info'
        with open(os.path.join(self.outpath, f'{title}.json'), 'w', encoding='utf-8') as file:
            json.dump(info, file, ensure_ascii=False, indent=4)

    def stats_import(self, translate=False):
        s_time = datetime.datetime.now()

        title = 'votes distributions'
        self.votes_dist_df = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'Meta voters'
        self.voters = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'votes unnormalized'
        self.votesdf = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig', index_col=0)

        title = 'Towns cluster'
        self.town_clusters = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'Raw data'
        self.rawdf = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig', index_col=0)

        title = 'town distances'
        self.distances = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'notes signs'
        with open(os.path.join(self.outpath, f'{title}.json'), encoding='utf-8-sig') as file:
            self.ballots = json.load(file)

        title = 'additional info'
        with open(os.path.join(self.outpath, f'{title}.json'), encoding='utf-8-sig') as file:
            info = json.load(file)
            self.metadata = info['metadata']
            self.parties = info['parties']
            self.k = info['k']
            non_empy_parties = info['non_empy_parties']
            self.non_empy_parties = pd.Series(index=non_empy_parties, data=True)

        if translate:
            path = os.path.join(self.datapath, r'parties_translation.json')
            if os.path.exists(path):
                with open(path, encoding='utf-8-sig') as file:
                    self.parties_names_translation = json.load(file)

            self.rawdf = self.rawdf.rename(columns=self.parties_names_translation)
            self.votes_dist_df = self.votes_dist_df.rename(columns=self.parties_names_translation)
            self.votesdf = self.votesdf.rename(columns=self.parties_names_translation)
            self.voters = self.voters.rename(columns=self.parties_names_translation)
            self.parties = [self.parties_names_translation.get(pt, pt) for pt in self.parties]
            self.ballots = {k: self.parties_names_translation.get(pt, pt) for k, pt in self.ballots.items()}

        e_time = datetime.datetime.now()
        d_time = e_time - s_time
        print(f'Data load completed. Time: {d_time}')

    def evaluate_campaign(self, src, trgt=None):
        absdf = self.votesdf.sort_values(by=src, ascending=False)
        absdf = absdf.set_index('uid')

        normdf = self.votes_dist_df.sort_values(by=src, ascending=False)
        normdf = normdf.set_index('uid')
        total_votes = float(normdf['Votes'].sum())

        normdf['Votes ratio'] = normdf['Votes'].astype(float) / total_votes

        min_val_for_attack = 0.01
        normdf = normdf[(normdf[src] > min_val_for_attack) & (normdf[trgt] > min_val_for_attack)]
        absdf = absdf.loc[normdf.index]

        normdf = normdf.sort_values(by='uid', ascending=False)
        absdf = absdf.sort_values(by='uid', ascending=False)

        seat_abs = total_votes / 120.0
        seat_norm = 1.0 / 120.0
        # Possible gain is 0.1 * min(|src|,|trgt|)

        abs_gain = pd.concat([absdf[src], absdf[trgt]], axis=1).min(axis=1)
        abs_gain = (0.1 * abs_gain).astype(int)
        norm_gain = abs_gain.astype(float) / total_votes
        normdf['Votes gain'] = abs_gain
        normdf['Votes ratio gain'] = norm_gain
        normdf['Seats gain'] = abs_gain.astype(float) / seat_abs

        normdf = normdf.sort_values(by='Votes gain', ascending=False)
        absdf = absdf.loc[normdf.index]

        normdf['seats sum'] = 1

        qdf = normdf.iloc[:10].copy()
        qdf['attacker'] = src
        qdf['target'] = trgt
        campaign_value = qdf['Seats gain'].sum()

        # print("-------------------------")
        # print(f"{src} --> {trgt}")
        # for ridx, r in qdf.iterrows():
        #     print(f"{r['town']}\tSeats: {100 * r['Seats gain']:>.1f}%")
        # print(f"Total: {100 * campaign_value:>.1f}%")
        # print("-------------------------")

        return campaign_value, qdf


if __name__ == '__main__':
    path_root = os.path.join(os.path.dirname(__file__), 'Dump')
    clear_folder(path_root)

    section_parameters = True
    if section_parameters:
        # Current state:
        govnt_prty = dict()
        govnt_prty['Likud'] = 36
        govnt_prty['Yesh Atid (Lapid)'] = 23
        govnt_prty['Religious Zionists'] = 10
        govnt_prty['Blue and White (Gantz)'] = 9
        govnt_prty['Shas'] = 9
        govnt_prty['United Torah Judaism'] = 7
        govnt_prty['Joint list'] = 6
        govnt_prty['Avoda'] = 6
        govnt_prty['Israel Beitenu'] = 6
        govnt_prty['New Hope (Saar)'] = 5
        govnt_prty['Raam (Abbas)'] = 4

        govnt_disagree = dict()
        govnt_disagree['Likud'] = ['New Hope (Saar)', 'Blue and White (Gantz)', 'Israel Beitenu', 'Yesh Atid (Lapid)',
                                   'Avoda', 'Meretz']
        govnt_disagree['Religious Zionists'] = ['Raam (Abbas)', 'Joint list']
        govnt_disagree['Yesh Atid (Lapid)'] = ['Shas', 'United Torah Judaism']

        # # New state:
        # new_prty = dict()
        # new_prty['Likud'] = 36
        # new_prty['Yesh Atid (Lapid)'] = 23
        # new_prty['Religious Zionists'] = 10
        # new_prty['Blue and White (Gantz)'] = 9
        # new_prty['Shas'] = 9
        # new_prty['United Torah Judaism'] = 7
        # new_prty['Joint list'] = 6
        # new_prty['Avoda'] = 6
        # new_prty['Israel Beitenu'] = 6
        # new_prty['New Hope (Saar)'] = 5
        # new_prty['Raam (Abbas)'] = 4
        #
        # new_disagree = dict()
        # new_disagree['Likud'] = ['New Hope (Saar)', 'Blue and White (Gantz)', 'Israel Beitenu', 'Yesh Atid (Lapid)',
        #                          'Avoda', 'Meretz']
        # new_disagree['Religious Zionists'] = ['Raam (Abbas)', 'Joint list']
        # new_disagree['Yesh Atid (Lapid)'] = ['Shas', 'United Torah Judaism']

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

    section_scan_all_moves = True
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
        print('Likely moves stored at: ' + os.path.join(path_root, 'moves.csv'))

    section_apply_specific_moves = False
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

if __name__ == '__main__':
    print("\n\n")
    print("END OF CODE.")
