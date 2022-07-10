import copy
import time

import numpy as np
import pandas as pd
from IPython.display import display
import itertools
import os
import shutil
from tqdm import tqdm


def clear_folder(path, clear_is_exists=True):
    if clear_is_exists and os.path.exists(path):
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
            if c_prty not in self.parties.keys():
                continue
            antiprtys = [cp for cp in self.parties.keys() if cp in antiprtys]

            if c_prty in self.coalitions_validity:
                prty_idx = self.coalitions_validity[c_prty].eq(1)
                cvdf = self.coalitions_validity[prty_idx]
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
    restrictions = base.disagree

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

            # if foe == 'new_hope':
            #     shap_t.to_csv(title='likud_attacked')

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
        if disagree_prty in ['new_hope', 'meretz']:
            shap_t.to_csv(title=f'{disagree_prty}_joined_likud')

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


if __name__ == '__main__':
    prty = dict()
    prty['likud'] = 36
    prty['yesh_atid'] = 23
    prty['tzionot_datit'] = 10
    prty['kahol_lavan'] = 9
    prty['shas'] = 9
    prty['yahadut_ha_tora'] = 7
    prty['meshutefet'] = 6
    prty['avoda'] = 6
    prty['israel_beitenu'] = 6
    prty['new_hope'] = 5
    prty['raam'] = 4

    disagree = dict()
    disagree['likud'] = ['new_hope', 'kahol_lavan', 'israel_beitenu', 'yesh_atid', 'avoda', 'meretz']
    disagree['tzionot_datit'] = ['raam', 'meshutefet']
    disagree['yesh_atid'] = ['shas', 'yahadut_ha_tora']

if __name__ == '__main__':
    path_root = os.path.join(os.path.dirname(__file__), 'Dump')
    clear_folder(path_root)

    shap = PoliticalShapley()
    shap.add_parties(prty)
    shap.add_restrictions(disagree)
    shap.run()
    print("----- Current board ----- ")
    df = shap.get_mandates()
    df.to_csv(os.path.join(path_root, 'Mandates.csv'))
    display(df)
    print("Mandates df saved to: " + os.path.join(path_root, 'Mandates.csv'))
    print("----- Current Shapley ----- ")
    df = pd.DataFrame(shap.get_shapley())
    df.to_csv(os.path.join(path_root, 'Shapley_vals.csv'))
    display(df)
    print("Current Shapley df saved to: " + os.path.join(path_root, 'Shapley_vals.csv'))
    print("----- Coalitions ----- ")
    df = shap.get_possible_govt()
    df.to_csv(os.path.join(path_root, 'Coalitions.csv'))
    display(df)
    print("Possible Coalitions df saved to: " + os.path.join(path_root, 'Coalitions.csv'))

    print("-------------------------")
    print("-------------------------")
    prty = 'israel_beitenu'
    prty_val = shap.get_shapley(prty)
    print(f"{prty}: {prty_val}")
    print(f"---------------------")
    print(f'Mandates: {shap.parties[prty]}')
    print(f"Shap: {prty_val}")
    print(f"---------------------")

    coals = shap.get_possible_govt()
    coals_with = 0
    coals_without = len(coals)
    if prty in coals.columns:
        coals_with = len(coals[coals[prty] > 0])
        coals_without = len(coals[coals[prty] <= 0])
    print(f'Coalitions with party: {coals_with}')
    print(f'Coalitions without party: {coals_without}')
    print("-------------------------")

    offense = get_campagin_tactics(shap, prty)
    time.sleep(0.2)
    print("Value of attack:")
    display(offense)

    # print("\n")
    # defence = get_weak_spots(shap, govnt_prty)
    # time.sleep(0.2)
    # print("Value of defence:")
    # display(defence)
    #
    # print("\n")
    # interest_to_attack = who_has_interest_to_attack(shap, govnt_prty)
    # time.sleep(0.2)
    # print("Likely to attack:")
    # display(interest_to_attack)

    # print("\n")
    # rule_to_break = which_rule_to_break(shap, 'likud')
    # time.sleep(0.2)
    # print("Most valueable partner for likud:")
    # display(rule_to_break)

    # df = shap.get_possible_govt()
    # if df.shape[0] > 0:
    #     display(df)
    # else:
    #     print("No possible coalition.")
    #
    # df = shap.shapley_values
    # print(df)
    # shap.to_csv(r'C:\school\PoliticalShapley')

    print("\n\n")
    print("END OF CODE.")
