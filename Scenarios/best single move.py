from src.fiddle import State
from src.fiddle import Voter
from src.fiddle import clear_folder

import os
import sys

from tqdm import tqdm

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(root_path, 'data')
    clear_folder(results_path, delete_if_exist=True)

if __name__ == '__main__':
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
    govnt_disagree['Likud'] = ['New Hope (Saar)', 'Israel Beitenu', 'Yesh Atid (Lapid)',
                               'Avoda', 'Meretz', 'Joint list']
    govnt_disagree['Religious Zionists'] = ['Raam (Abbas)', 'Joint list']

    # govnt_disagree['Yesh Atid (Lapid)'] = ['Shas', 'United Torah Judaism']
    govnt_disagree['Meretz'] = ['Shas', 'United Torah Judaism']
    govnt_disagree['Avoda'] = ['Shas', 'United Torah Judaism']

if __name__ == '__main__':
    player = 'Religious Zionists'

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
        root_state.get_possible_govt().to_csv(os.path.join(results_path, 'base possible_govt.csv'))
        root_state.get_shapley().to_csv(os.path.join(results_path, 'base power.csv'))

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
            msg += f'SRC ({src}) power [{curr_shaps[src]:.1f}] --> [{new_shap[src]:.1f}] ({shap_delta[src]:>.1f})' + '\n'
            msg += f'TRGT ({trgt}) power [{curr_shaps[trgt]:.1f}] --> [{new_shap[trgt]:.1f}] ({shap_delta[trgt]:>.1f})' + '\n'

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

    section_actions_to_look_out_for = True
    if section_actions_to_look_out_for:
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

            res = dict()
            res['token'] = action_tuple
            res['SRC'] = src
            res['TRGT'] = trgt
            res['ACTION'] = action
            res[f'POWER GAIN FOR {player}'] = shap_delta[player]

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

if __name__ == '__main__':
    print("END OF CODE.")
