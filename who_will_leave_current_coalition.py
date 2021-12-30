import pandas as pd

from PoliticalShapley import *

if __name__ == '__main__':
    path_root = os.path.join(os.path.dirname(__file__), 'Dump')
    clear_folder(path_root)

    section_parameters = True
    if section_parameters:
        # Current state:
        current_prty = dict()
        current_prty['likud'] = 30
        current_prty['new_hope'] = 6
        current_prty['yamina'] = 7
        current_prty['yesh_atid'] = 17
        current_prty['meshutefet'] = 6
        current_prty['shas'] = 9
        current_prty['yahadut_ha_tora'] = 7
        current_prty['israel_beitenu'] = 7
        current_prty['tzionot_datit'] = 6
        current_prty['meretz'] = 6
        current_prty['avoda'] = 7
        current_prty['kahol_lavan'] = 8
        current_prty['raam'] = 4

        current_disagree = dict()
        current_disagree['likud'] = ['new_hope', 'kahol_lavan', 'israel_beitenu', 'yesh_atid', 'avoda', 'meretz']
        current_disagree['meshutefet'] = ['likud', 'tzionot_datit']
        current_disagree['yesh_atid'] = ['shas', 'yahadut_ha_tora']

        # New state:
        new_prty = dict()
        new_prty['likud'] = 34
        new_prty['new_hope'] = 0
        new_prty['yamina'] = 6
        new_prty['yesh_atid'] = 19
        new_prty['meshutefet'] = 6
        new_prty['shas'] = 9
        new_prty['yahadut_ha_tora'] = 7
        new_prty['israel_beitenu'] = 6
        new_prty['tzionot_datit'] = 7
        new_prty['meretz'] = 5
        new_prty['avoda'] = 7
        new_prty['kahol_lavan'] = 9
        new_prty['raam'] = 5

        new_disagree = dict()
        new_disagree['likud'] = ['new_hope', 'kahol_lavan', 'yesh_atid', 'avoda', 'meretz', 'israel_beitenu']
        new_disagree['meshutefet'] = ['likud', 'tzionot_datit']
        new_disagree['yesh_atid'] = ['shas', 'yahadut_ha_tora']

        current_govt = ['new_hope', 'yamina', 'yesh_atid', 'israel_beitenu', 'meretz', 'avoda', 'kahol_lavan', 'raam']
        current_opos = [k for k in new_prty.keys() if k not in current_govt]

    section_base_mode = True
    if section_base_mode:
        curr_shap = PoliticalShapley()
        curr_shap.add_parties(current_prty)
        curr_shap.add_restrictions(current_disagree)
        curr_shap.run()

        new_shap = PoliticalShapley()
        new_shap.add_parties(new_prty)
        new_shap.add_restrictions(new_disagree)
        new_shap.run()

        mapdf = curr_shap.get_possible_govt()
        mapdf.replace(0, '').to_csv(os.path.join(path_root, 'Current_map.csv'))
        mapdf = new_shap.get_possible_govt()
        mapdf.replace(0, '').to_csv(os.path.join(path_root, 'New_map.csv'))

        print("----- Seats ----- ")
        seatsdf = pd.DataFrame(columns=['in current govt', 'Current', 'Potential', 'Delta'],
                               index=list(new_prty.keys()))
        seatsdf.loc[current_govt, 'in current govt'] = 1
        seatsdf['Current'] = curr_shap.get_mandates()['Mandates']
        seatsdf['Potential'] = new_shap.get_mandates()['Mandates']
        seatsdf = seatsdf.fillna(0).astype(int)
        seatsdf['Delta'] = seatsdf['Potential'] - seatsdf['Current']

        seatsdf.to_csv(os.path.join(path_root, 'seats.csv'))
        display(seatsdf)

        print("----- Leverage ----- ")
        shapdf = pd.DataFrame(columns=['in current govt', 'Current', 'Potential', 'Delta'], index=list(new_prty.keys()))
        shapdf.loc[current_govt, 'in current govt'] = 1
        shapdf['Current'] = curr_shap.get_shapley()
        shapdf.loc[current_opos, 'Current'] = 0
        shapdf['Potential'] = new_shap.get_shapley()
        shapdf = shapdf.fillna(0).astype(float)
        shapdf['Delta'] = shapdf['Potential'] - shapdf['Current']
        shapdf = shapdf.sort_values(by=['Delta'], ascending=False)

        shapdf.to_csv(os.path.join(path_root, 'shaps.csv'))
        shapdf = shapdf.loc[current_govt].sort_values(by=['Delta'], ascending=False)
        shapdf.to_csv(os.path.join(path_root, 'shaps_only_govt.csv'))
        display(shapdf)
