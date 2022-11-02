from src.fiddle import State
from src.fiddle import Voter
from src.fiddle import clear_folder

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import sys

from tqdm import tqdm
import json

from tabulate import tabulate

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(root_path, 'data')
    clear_folder(results_path, delete_if_exist=True)

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(root_path), 'coalitions_data')
    game_path = os.path.join(data_path, 'news_14.json')
    rules_path = os.path.join(data_path, 'rules.json')

    if not os.path.exists(game_path):
        prty = dict()
        prty['likud'] = 31
        prty['yesh_atid'] = 25
        prty['tzionot_datit'] = 14
        prty['Mahane_Mamlachti'] = 11
        prty['shas'] = 8
        prty['yahadut_ha_tora'] = 7
        prty['israel_beitenu'] = 6
        prty['avoda'] = 5
        prty['meretz'] = 5
        prty['Hadash_Taal'] = 4
        prty['raam'] = 4

        with open(game_path, 'w') as f:
            json.dump(prty, f, indent=4)

    else:
        with open(game_path, 'r') as f:
            prty = json.load(f)

    reload_rules = False
    if (not reload_rules) or not os.path.exists(rules_path):
        disagree = dict()
        disagree['likud'] = ['yesh_atid', 'avoda', 'meretz', 'Hadash_Taal']
        disagree['tzionot_datit'] = ['raam', 'Hadash_Taal']
        disagree['meretz'] = ['tzionot_datit']
        disagree['yesh_atid'] =  ['shas', 'yahadut_ha_tora']
        disagree['israel_beitenu'] = ['shas', 'yahadut_ha_tora']
        # disagree['Mahane_Mamlachti'] = ['tzionot_datit']
        disagree['Hadash_Taal'] = ['yesh_atid', 'likud']

        with open(rules_path, 'w') as f:
            json.dump(disagree, f, indent=4)
    else:
        with open(rules_path, 'r') as f:
            disagree = json.load(f)

if __name__ == '__main__':
    # Current state:
    govnt_prty = prty
    govnt_disagree = disagree

if __name__ == '__main__':
    player = 'Likud'

    root_state = State()
    campaign_value = 2
    root_state = State(campaign_value=campaign_value)
    root_state.add_parties(govnt_prty)
    root_state.add_voter()
    root_state.add_restrictions(govnt_disagree)
    root_state.run()
    curr_shaps = root_state.get_shapley()

    section_base_mode = True
    if section_base_mode:
        pos_df = root_state.get_possible_govt()
        pos_df.to_csv(os.path.join(results_path, 'base possible_govt.csv'))
        powerdf = root_state.get_shapley()
        powerdf.to_csv(os.path.join(results_path, 'base power.csv'))
        print('Seats:')
        seatsdf = pd.DataFrame(prty.items()).sort_values(1, ascending=False)
        print(tabulate(seatsdf, headers='keys', tablefmt='psql'))

        print('base power:')
        print(tabulate(pd.DataFrame(powerdf), headers='keys', tablefmt='psql'))
        print('base possible_govt:')
        print(tabulate(pos_df, headers='keys', tablefmt='psql'))

        print(f'Results saved to {results_path}')

    section_next_move_values = False
    if section_next_move_values:
        actions = root_state.get_actions(player)

        results = list()
        for action_tuple in tqdm(actions, total=len(actions)):
            action, src, trgt = action_tuple
            q = root_state.evaluate_campaign_by_voters(src, trgt)
            new_state = root_state.apply(action=action_tuple)
            new_shap = new_state.get_shapley()
            shap_delta = new_shap - curr_shaps

            if action == 'Campaign':
                gain, towns = root_state.evaluate_campaign_by_voters(src, trgt)

            res = dict()
            res['token'] = action_tuple
            res['SRC'] = src
            res['TRGT'] = trgt
            res['ACTION'] = action
            res['POWER GAIN'] = shap_delta[src]

            msg = '------------------------------------'
            msg += '\n'
            msg += f'{src} [{action}] {trgt}\n'
            msg += f'SRC ({src}) power [{curr_shaps.get(src, 0):.1f}] --> [{new_shap.get(src, 0):.1f}] ({shap_delta.get(src, 0):>.1f})' + '\n'
            msg += f'TRGT ({trgt}) power [{curr_shaps.get(trgt, 0):.1f}] --> [{new_shap.get(trgt, 0):.1f}] ({shap_delta.get(trgt, 0):>.1f})' + '\n'

            if action == 'Campaign':
                msg += f'Seats: {gain:>.3f}' + '\n'
                msg += f'Cities:\n'
                msg += ', '.join(towns['town'].head(5).values) + '\n'
                res['towns'] = towns

            res['msg'] = msg

            res['power df'] = new_state.get_shapley()
            res['possible governments df'] = new_state.get_possible_govt()

            results.append(res)

        results.sort(key=lambda d: d['POWER GAIN'], reverse=True)
        for idx, d in enumerate(results):
            token = d['token']
            power_gain = d['POWER GAIN']
            if power_gain <= 0:
                continue

            title = f'{idx + 1} {token[0]} {token[1]} {token[2]}'
            folder_path = os.path.join(results_path, title)
            clear_folder(folder_path, delete_if_exist=True)

            with open(os.path.join(folder_path, title + ' results.txt'), 'w', encoding="utf-8-sig") as f:
                f.write(d['msg'])
            d['power df'].to_csv(os.path.join(folder_path, title + ' power.csv'))
            d['possible governments df'].to_csv(os.path.join(folder_path, title + ' possible governments.csv'))
            if 'towns' in d:
                d['towns'].to_csv(os.path.join(folder_path, title + ' towns.csv'),
                                  encoding="utf-8-sig",
                                  )

            print(d['msg'])
            print(f"Exported to: {folder_path}")

        subsection_summerize_csv = True
        if subsection_summerize_csv:
            df = pd.DataFrame(results)
            df = df[['token', 'SRC', 'TRGT', 'ACTION', 'POWER GAIN']]
            df.to_csv(os.path.join(results_path, 'optimal moves summary.csv'), encoding="utf-8-sig")
            print(f"Exported to: {results_path}")

    section_actions_that_will_do_the_most_harm = False
    if section_actions_that_will_do_the_most_harm:
        actions = root_state.get_actions_poiter_at(player)

        results = list()
        for action_tuple in tqdm(actions, total=len(actions)):
            action, src, trgt = action_tuple
            q = root_state.evaluate_campaign_by_voters(src, trgt)
            new_state = root_state.apply(action=action_tuple)
            new_shap = new_state.get_shapley()
            shap_delta = new_shap - curr_shaps

            if action == 'Campaign':
                gain, towns = root_state.evaluate_campaign_by_voters(src, trgt)

            if player not in new_shap:
                continue
            if trgt not in new_shap:
                continue
            if src not in new_shap:
                continue

            res = dict()
            res['token'] = action_tuple
            res['SRC'] = src
            res['TRGT'] = trgt
            res['ACTION'] = action
            res[f'POWER GAIN FOR {player}'] = shap_delta[player]
            res[f'POWER GAIN FOR Actor'] = shap_delta[src]

            msg = '------------------------------------'
            msg += '\n'
            msg += f'{src} [{action}] {trgt}\n'
            msg += f'SRC ({src}) power [{curr_shaps.get(src, 0):.1f}] --> [{new_shap.get(src, 0):.1f}] ({shap_delta.get(src, 0):>.1f})' + '\n'
            msg += f'TRGT ({trgt}) power [{curr_shaps.get(trgt, 0):.1f}] --> [{new_shap.get(trgt, 0):.1f}] ({shap_delta.get(trgt, 0):>.1f})' + '\n'
            msg += f'<> {player} [{curr_shaps[player]:.1f}] --> [{new_shap[player]:.1f}] ({shap_delta[player]:>.1f})' + '\n'

            if action == 'Campaign':
                msg += f'Seats: {gain:>.3f}' + '\n'
                msg += f'Cities:\n'
                msg += ', '.join(towns['town'].head(5).values) + '\n'
                res['towns'] = towns

            res['msg'] = msg

            res['power df'] = new_state.get_shapley()
            res['possible governments df'] = new_state.get_possible_govt()

            results.append(res)

        results.sort(key=lambda d: d[f'POWER GAIN FOR {player}'], reverse=False)
        for idx, d in enumerate(results):
            token = d['token']
            power_gain = d[f'POWER GAIN FOR Actor']
            if power_gain <= 0:
                continue
            title = f'{idx + 1} blind-Spot {token[0]} {token[1]} {token[2]}'
            folder_path = os.path.join(results_path, title)
            clear_folder(folder_path, delete_if_exist=True)

            with open(os.path.join(folder_path, title + ' results.txt'), 'w', encoding="utf-8-sig") as f:
                f.write(d['msg'])
            d['power df'].to_csv(os.path.join(folder_path, title + ' power drop.csv'))
            d['possible governments df'].to_csv(os.path.join(folder_path, title + ' possible governments.csv'))
            if 'towns' in d:
                d['towns'].to_csv(os.path.join(folder_path, title + ' towns.csv'),
                                  encoding="utf-8-sig",
                                  )

            print(d['msg'])
            print(f"Exported to: {folder_path}")
            subsection_summerize_csv = True
            if subsection_summerize_csv:
                df = pd.DataFrame(results)
                df = df[['token', 'SRC', 'TRGT', 'ACTION', f'POWER GAIN FOR {player}', f'POWER GAIN FOR Actor']]
                df.to_csv(os.path.join(results_path, 'vulnerability summary.csv'), encoding="utf-8-sig")
                print(f"Exported to: {results_path}")

if __name__ == '__main__':
    print("END OF CODE.")
