import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os


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
        self.legal_coalitions = None

    def add_parties(self, parties):
        self.parties.update(parties)

    def add_restrictions(self, restrictions):
        self.disagree.update(restrictions)

    def run(self):
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
            prty_idx = self.coalitions_validity[c_prty].eq(1)
            for antiprty in antiprtys:
                antiprty_idx = self.coalitions_validity[antiprty].eq(1)
                self.coalitions_validity.loc[prty_idx & antiprty_idx, 'invalid_count'] += 1

        self.coalitions_validity['valid'] = self.coalitions_validity['invalid_count'].eq(0)

        self.coalitions_value['value'] = 0
        good_coalitions = self.coalitions_validity['valid'] & (self.coalitions_mandats['Total'] > 60)
        self.coalitions_value.loc[good_coalitions, 'value'] = 100

        # Calculate shapley
        self.shapley_values = pd.Series(index=self.parties.keys())
        for c_prty in self.parties.keys():
            other_parties = [cp for cp in self.parties.keys() if cp != c_prty]
            withdf = self.coalitions_value[self.coalitions_value[c_prty].eq(1)]
            withoutdf = self.coalitions_value[self.coalitions_value[c_prty].eq(0)]
            reordered = pd.concat([withdf, withoutdf])
            reordered = reordered.sort_values(by=other_parties)
            reordered['gain'] = (reordered['value'] - reordered['value'].shift(-1)).fillna(0).astype(int).clip(0)
            reordered = reordered.loc[self.coalitions_value[c_prty].eq(1), [c for c in reordered.columns
                                                                            if c != 'value']]
            reordered['N'] = len(self.parties)
            reordered['S'] = reordered[other_parties].sum(axis=1)

            n_fac = np.math.factorial(len(self.parties))
            s_fac = reordered['S'].apply(np.math.factorial)
            comp_s_fac = (reordered['N'] - reordered['S'] - 1).apply(np.math.factorial)
            gains = reordered['gain']
            shap_gain = (s_fac * comp_s_fac).astype(float) / float(n_fac) * gains
            shap_val = shap_gain.sum()
            self.shapley_values[c_prty] = shap_val
        self.shapley_values = self.shapley_values.sort_values(ascending=False)
        self.legal_coalitions = self.coalitions_mandats.loc[self.coalitions_value['value'] > 0].reset_index()

    def get_possible_govt(self):
        return self.legal_coalitions

    def get_shapley(self):
        return self.shapley_values

    def to_csv(self, path=None):

        shap_path = os.path.join(path, 'Shapley.csv')
        self.shapley_values.to_csv(shap_path, header=True)

        possible_govt_path = os.path.join(path, 'possible options.csv')
        self.legal_coalitions.to_csv(possible_govt_path)


if __name__ == '__main__':
    prty = dict()
    prty['likud'] = 29
    prty['new_hope'] = 14
    prty['yamina'] = 13
    prty['yesh_atid'] = 18
    prty['meshutefet'] = 9
    prty['shas'] = 8
    prty['yahadut_ha_tora'] = 7
    prty['israel_beitenu'] = 6
    prty['tzionot_datit'] = 5
    # prty['meretz'] = 0
    prty['avoda'] = 7
    prty['kahol_lavan'] = 4

    disagree = dict()
    disagree['likud'] = ['new_hope', 'kahol_lavan', 'israel_beitenu', 'yesh_atid', 'avoda']
    disagree['meshutefet'] = ['likud', 'tzionot_datit', 'yamina']
    disagree['yesh_atid'] = ['shas', 'yahadut_ha_tora']

if __name__ == '__main__':
    shap = PoliticalShapley()
    shap.add_parties(prty)
    shap.add_restrictions(disagree)
    shap.run()

    shap.to_csv(r'C:\school\PoliticalShapley')
