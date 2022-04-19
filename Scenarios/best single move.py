from src.fiddle import State
from src.fiddle import Voter

from tqdm import tqdm

if __name__ == '__main__':
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

if __name__ == '__main__':
    player = 'Avoda'

    root_state = State()
    campaign_value = 2
    root_state = State(campaign_value=campaign_value)
    root_state.add_parties(govnt_prty)
    root_state.add_voter()
    root_state.add_restrictions(govnt_disagree)
    root_state.run()
    curr_shaps = root_state.get_shapley()

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
            j = 3

        res['msg'] = msg
        results.append(res)

    results.sort(key=lambda d: d['POWER GAIN'])
    for d in results:
        print(d['msg'])

if __name__ == '__main__':
    print("END OF CODE.")
