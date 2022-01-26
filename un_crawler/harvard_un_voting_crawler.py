#!/usr/bin/env python

## Overview
# Extracting information from UNBisNET Voting Records (http://unbisnet.un.org:8080/ipac20/ipac.jsp?profile=)
# Data sheets with voting records were copied to local drive using Scrapy in Python
# Individual votes get coded as (http://research.un.org/en/docs/ga/voting):
# YES, NO, ABSTAIN, ABSENT (coded as blank space), INELIGIBLE (based on code 9 in the UNBisNET data base)
# Missing indicates that the country was not a UN member/did not exist at the time of voting
# Output is delimted text-file (delimiter: semi-colon), some fields (title, documents) are enclosed in double-quotation marks

## Comments
# The original source included some typos (e.g., codes for votes: NY, 9, 90), these were addressed by manually
# checking the meeting records and adding the correct information
# Country might not exist at that moment in time (e.g., Czech Republic), this is coded as missing
# No positive identification of absence (which is denoted only by a blank space)

import os
import sys
import re
from bs4 import BeautifulSoup


def clear_folder(path, delete_if_exist=True):
    if os.path.exists(path) and delete_if_exist:
        all_items_to_remove = [os.path.join(path, f) for f in os.listdir(path)]
        for item_to_remove in all_items_to_remove:
            if os.path.exists(item_to_remove) and not os.path.isdir(item_to_remove):
                os.remove(item_to_remove)
            else:
                shutil.rmtree(item_to_remove)

    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    root_path = os.path.dirname(__file__)
    outdir = os.path.join(root_path, 'data')
    ouput_rc_filename = os.path.join(outdir, 'Votes.txt')
    ouput_missing_filename = os.path.join(outdir, 'Missing.txt')
    ouput_typo_filename = os.path.join(outdir, 'Typo.txt')

    clear_folder(outdir, False)
    clear_folder(outdir, True)


countries = (
    "AFGHANISTAN", "ALBANIA", "ALGERIA", "ANDORRA", "ANGOLA", "ANTIGUA AND BARBUDA", "ARGENTINA", "ARMENIA",
    "AUSTRALIA",
    "AUSTRIA", "AZERBAIJAN", "BAHAMAS", "BAHRAIN", "BANGLADESH", "BARBADOS", "BELARUS", "BELGIUM", "BELIZE", "BENIN",
    "BHUTAN", "BOLIVIA (PLURINATIONAL STATE OF)", "BOLIVIA", "BOSNIA AND HERZEGOVINA", "BOSNIA", "BOTSWANA", "BRAZIL",
    "BRUNEI DARUSSALAM", "BULGARIA", "BURKINFASO", "BURKINA FASO", "BURMA", "BURUNDI", "BYELORUSSIAN SSR", "CAPE VERDE",
    "CABO VERDE", "CAMBODIA", "CAMEROON", "CANADA", "CENTRAL AFRICAN REPUBLIC", "CENTRAL AFRICAN EMPIRE", "CEYLON",
    "CHAD",
    "CHILE", "CHINA", "COLOMBIA", "COMOROS", "CONGO (BRAZZAVILLE)", "CONGO (LEOPOLDVILLE)", "CONGO", "COSTA RICA",
    "COTE D'IVOIRE", "CROATIA", "CUBA", "CYPRUS", "CZECHOSLOVAKIA", "CZECH REPUBLIC", "DAHOMEY",
    "DEMOCRATIC PEOPLE'S REPUBLIC OF KOREA", "DEMOCRATIC REPUBLIC OF THE CONGO", "DENMARK", "DJIBOUTI",
    "DOMINICAN REPUBLIC", "DOMINICA", "EAST-TIMOR", "ECUADOR", "EGYPT", "EL SALVADOR", "EQUATORIAL GUINEA", "ERITREA",
    "ESTONIA", "ETHIOPIA", "FEDERATION OF MALAYA", "FIJI", "FINLAND", "FRANCE", "GABON", "GAMBIA", "GEORGIA",
    "GERMAN DEMOCRATIC REPUBLIC", "GERMANY, FEDERAL REPUBLIC OF", "GERMANY", "GHANA", "GREECE", "GRENADA", "GUATEMALA",
    "GUINEA-BISSAU", "GUINEA", "GUYANA", "HAITI", "HONDURAS", "HUNGARY", "ICELAND", "INDIA", "INDONESIA",
    "IRAN (ISLAMIC REPUBLIC OF)", "IRAN", "IRAQ", "IRELAND", "ISRAEL", "ITALY", "IVORY COAST", "JAMAICA", "JAPAN",
    "JORDAN",
    "KAZAKHSTAN", "KENYA", "KAMPUCHEA", "KIRIBATI", "REPUBLIC OF KIRIBATI", "KHMER REPUBLIC", "KUWAIT", "KYRGYZSTAN",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC", "LATVIA", "LAOS", "LEBANON", "LESOTHO", "LIBERIA", "LIBYAN ARAB JAMAHIRIYA",
    "LIBYAN ARAB REPUBLIC", "LIBYA", "LIBYAN ARAB REPUBLIC", "LIECHTENSTEIN", "LITHUANIA", "LUXEMBOURG", "MADAGASCAR",
    "MALAWI", "MALAYSIA", "MALDIVE ISLANDS", "MALDIVES", "MALI", "MALTA", "MARSHALL ISLANDS", "MAURITANIA", "MAURITIUS",
    "MEXICO", "MICRONESIA (FEDERATED STATES OF)", "MONACO", "MONGOLIA", "MONTENEGRO", "MOROCCO", "MOZAMBIQUE",
    "MYANMAR",
    "NAMIBIA", "NAURU", "NEPAL", "NETHERLANDS", "NEW ZEALAND", "NICARAGUA", "NIGER", "NIGERIA", "NORWAY", "OMAN",
    "PAKISTAN", "PALAU", "PANAMA", "PAPUA NEW GUINEA", "PARAGUAY", "PERU", "PHILIPPINE REPUBLIC", "PHILIPPINES",
    "POLAND",
    "PORTUGAL", "QATAR", "REPUBLIC OF KOREA", "REPUBLIC OF MOLDOVA", "MOLDOVA", "RHODESIA", "ROMANIA",
    "RUSSIAN FEDERATION",
    "RWANDA", "SAINT CHRISTOPHER AND NEVIS", "SAINT KITTS AND NEVIS", "SAINT LUCIA", "SAINT VINCENT AND THE GRENADINES",
    "SAMOA", "SAN MARINO", "SAO TOME AND PRINCIPE", "SAUDI ARABIA", "SENEGAL", "SERBIA", "SEYCHELLES", "SIAM",
    "SIERRA LEONE", "SINGAPORE", "SLOVAKIA", "SLOVENIA", "SOLOMON ISLANDS", "SOMALIA", "UNION OF SOUTH AFRICA",
    "SOUTH AFRICA", "SOUTHERN YEMEN", "SOUTH SUDAN", "SPAIN", "SRI LANKA", "SUDAN", "SURINAME", "SWAZILAND", "SWEDEN",
    "SWITZERLAND", "SYRIAN ARAB REPUBLIC", "SYRIA", "TAJIKISTAN", "THAILAND",
    "THE FORMER YUGOSLAV REPUBLIC OF MACEDONIA",
    "TANGANYIKA", "TIMOR-LESTE", "TOGO", "TONGA", "TRINIDAD AND TOBAGO", "TUNISIA", "TURKEY", "TURKMENISTAN", "TUVALU",
    "UGANDA", "UKRAINE", "UKRAINIAN SSR", "UNITED ARAB EMIRATES", "UNITED ARAB REPUBLIC", "UNITED KINGDOM",
    "UNITED REPUBLIC OF TANZANIA", "UNITED REPUBLIC OF CAMEROON", "UNITED STATES", "UPPER VOLTA", "URUGUAY", "USSR",
    "UZBEKISTAN", "VANUATU", "VENEZUELA (BOLIVARIAN REPUBLIC OF)", "VENEZUELA", "VIET NAM", "YEMEN", "YUGOSLAVIA",
    "ZAIRE",
    "ZAMBIA", "ZANZIBAR", "ZIMBABWE")
# Lists of words to be ignored for the check for missing countries (e.g., name elements)
ignore = ("A", "Y", "N", "CAPE", "VERDE", "BRUNEI", "DARUSSALAM", "VIET", "NAM", "ARAB", "UNITED", "EMIRATES", "FORMER",
          "ISLANDS", "MARSHALL", "SAINT", "SAO", "SAN", "TOME", "PRINCIPE", "EQUATORIAL", "MICRONESIA", "MACEDONIA",
          "SAINT", "KITTS", "AND", "NEVIS", "LUCIA", "VINCENT", "THE", "GRENADINES", "RUSSIAN", "FEDERATION", "COSTA",
          "RICA", "CENTRAL", "PAPUA", "NEW", "GUINEA", "ANTIGUA", "BARBUDA", "SOLOMON", "KINGDOM", "ZEALAND",
          "PLURINATIONAL", "SRI", "LANKA", "LAO", "YUGOSLAV", "REPUBLIC", "UPPER", "VOLTA", "IVORY", "COAST",
          "TRINIDAD", "TOBAGO", "UKRAINIAN", "CZECH", "REPUBLIC", "SIERRA", "LEONE", "SAN", "MARINO", "BURKINA", "FASO",
          "EL", "SALVADOR", "TANZANIA", "FEDERAL", "DEMOCRATIC", "GERMAN", "SAUDI", "ARABIA", "SOUTH", "SUDAN", "COTE",
          "AFRICAN", "KOREA", "AFRICA", "MOLDOVA", "STATES", "HERZEGOVINA", "STATE", "OF", "CABO", "SYRIAN",
          "DOMINICAN", "LIBYAN", "JAMAHIRIYA", "SSR", "MOLDOVA", "SOUTHERN", "PHILIPPINE", "CHRISTOPHER",
          "BYELORUSSIAN", "EMPIRE", "MALDIVE", "MALAYA", "UNION", "KHMER")

# Lists for missing data/report
missing_date_list = []
missing_year_list = []
missing_res_number_list = []
missing_yes_vote_list = []
missing_no_vote_list = []
missing_abstentions_list = []
missing_absent_list = []
missing_total_list = []
missing_title_list = []
missing_countries = ()
missing_countries = set(missing_countries)
possible_typos = []

# Create output file and print first row with variable names
with open(ouput_rc_filename, "w") as output_rc:
    header = "rc_id" + ";" + "data sheet" + ";" + "res_number" + ";" + "title" + ";" + "document" + ";" + "meeting" + ";" + "vote_date" + ";" + "year" + ";" + "yes_votes" + ";" + "no_votes" + ";" + "abstentions" + ";" + "absent" + ";" + "total"
    countries_str = ';'.join(sorted(countries))
    msg = header + countries_str + '\n'
    output_rc.write(msg)
    output_rc.close()

# ## MAIN PART
# count = 1
# for data_sheet in data_sheets:
#     filename_data_sheet = path + data_sheet
#     current_file = open(filename_data_sheet)
#     # current_file=open("M:/Userdata/Current/Projects/UN Voting Behaviour/Webscraping/Data/478211.html")
#     soup = BeautifulSoup(current_file)
#     current_file.close()
#     voting_record = {}
#     for country in countries:
#         voting_record[country] = "Missing"
#     text = soup.get_text()
#     text = str(text.encode('ascii', 'ignore'))
#     search_string = str(re.findall(r'MARC\sDisplay(.*?)Copy', text, re.DOTALL))
#     # Correcting for slightly different spelling of country names in documents
#     search_string = search_string.replace("SURINAM", "SURINAME")
#     search_string = search_string.replace("SURINAMEE", "SURINAME")
#     search_string = search_string.replace("KYRGYZTAN", "KYRGYZSTAN")
#     search_string = search_string.replace("BURMNA", "BURMA")
#     # output_check=open("M:\Userdata\Current\Projects\UN Voting Behaviour\Webscraping\output_2.txt", "w")
#     # output_check.write(str(search_string))
#     # output_check.close()
#     # Checking for possibly missing country names in the data sheet
#     for country in countries:
#         search_string = search_string.replace(country, str(country) + "  ")
#     # Three white spaces are now in country names that contain a (former) country name (e.g. Bosnia and Herzegovina)
#     search_string = search_string.replace("   ", " ")
#     # SOMALIA and ROMANIA are affected by former operation despite being in the list and no entry "ROMAN" or "SOMALI" (why???)
#     search_string = search_string.replace("SOMALI  A", "SOMALIA  ")
#     search_string = search_string.replace("ROMAN  IA", "ROMANIA  ")
#     search_string = search_string.replace("NIGER  IA", "NIGERIA  ")
#     search_string = search_string.replace("SYRIA  N ARAB REPUBLIC", "SYRIAN ARAB REPUBLIC  ")
#     search_string = search_string.replace("DOMINICA  N REPUBLIC", "DOMINICAN REPUBLIC  ")
#     search_string = search_string.replace("GUINEA  -BISSAU", "GUINEA-BISSAU  ")
#     search_string = search_string.replace("GERMANY  , FEDERAL REPUBLIC OF", "GERMANY, FEDERAL REPUBLIC OF  ")
#     search_string = search_string.replace("LIBYA  N ARAB JAMAHIRIYA", "LIBYAN ARAB JAMAHIRIYA  ")
#     search_string = search_string.replace("LIBYA  N ARAB REPUBLIC", "LIBYAN ARAB REPUBLIC  ")
#     search_string_rc = str(re.findall(r'Detailed\sVoting:(.*)', search_string))
#     missing_country = re.findall(r"\s([A-Z]*?)\s", search_string_rc)
#     missing_country = [i for i in missing_country if i not in countries]
#     missing_country = [i for i in missing_country if i not in ignore]
#     missing_country = filter(None, missing_country)
#     sub = re.search(r"UN\sResolution\sSymbol:(.*?)Link", search_string)
#     if sub:
#         res_number = sub.group(1)
#     else:
#         res_number = "Missing"
#         # In some instances there is no entry for link, which requires a different regular expression to extract the resolution number
#     if res_number == "Missing":
#         sub = re.search(r"UN\sResolution\sSymbol:(.*?)Meeting", search_string)
#         if sub:
#             res_number = sub.group(1)
#
#     if missing_country:
#         output_missing = open(output_missing_filename, "a")
#         print >> output_missing, str(res_number) + ": " + str(missing_country)
#         output_missing.close()
#     missing_countries |= set(missing_country)
#
#     ## Extracting Information about the resolution
#     # Resolution number already extracted for missings
#     sub = re.search(r"Title:(.*?)\s:\sresolution", search_string)
#     if sub:
#         title = sub.group(1)
#     else:
#         title = "Missing"
#     # Sometimes, there is no statement "resolution/adopted by...
#     if title == "Missing":
#         sub = re.search(r"Title:(.*?)Related", search_string)
#         if sub:
#             title = sub.group(1)
#     sub = re.search(r"Related\sDocument:(.*?)Vote", search_string)
#     if sub:
#         document = sub.group(1)
#     else:
#         document = "Missing"
#
#     # Extracting date and re-formatting
#     sub = re.search(r"Vote\sDate:([0-9]{8})Agenda", search_string)
#     if sub:
#         date_string = str(sub.group(1))
#         if len(date_string) == 8:
#             year = date_string[0:4]
#             month = date_string[4:6]
#             month = month.lstrip("0")
#             day = date_string[-2:]
#             day = day.lstrip("0")
#             vote_date = day + "." + month + "." + year
#         else:
#             vote_date = "Missing"
#     else:
#         vote_date = "Missing"
#     # Sometimes there is not entry for Agenda and the next point is Detailed Voting, requiring a different regular expression to extract the date
#     if vote_date == "Missing":
#         sub = re.search(r"Vote\sDate:([0-9]{8})Detailed", search_string)
#         if sub:
#             date_string = str(sub.group(1))
#             if len(date_string) == 8:
#                 year = date_string[0:4]
#                 month = date_string[4:6]
#                 month = month.lstrip("0")
#                 day = date_string[-2:]
#                 day = day.lstrip("0")
#                 vote_date = day + "." + month + "." + year
#
#     sub = re.search(r"Meeting\sSymbol:(.*?)Title", search_string)
#     if sub:
#         meeting = sub.group(1)
#     else:
#         meeting = "Missing"
#     sub = re.search(r"Yes:\s*([0-9]*?)\s*,", search_string)
#     if sub:
#         yes_votes = sub.group(1)
#     else:
#         yes_votes = "Missing"
#     sub = re.search(r"No:\s*([0-9]*?)\s*,", search_string)
#     if sub:
#         no_votes = sub.group(1)
#     else:
#         no_votes = "Missing"
#     sub = re.search(r"Abstentions:\s*([0-9]*?)\s*,", search_string)
#     if sub:
#         abstentions = sub.group(1)
#     else:
#         abstentions = "Missing"
#     sub = re.search(r"Non-Voting:\s*([0-9]*?)\s*,", search_string)
#     if sub:
#         absent = sub.group(1)
#     else:
#         absent = "Missing"
#     sub = re.search(r"membership:\s*([0-9]*?)Vote", search_string)
#     if sub:
#         total = sub.group(1)
#     else:
#         total = "Missing"
#
#     # Extracting information on votes for each country
#     for country in countries:
#         # Country might not exist at that moment in time (e.g., Czech Republic), this is coded as missing
#         # No positive identification of absence (which is denoted only by a blank space)
#         my_regex = country
#         if re.search(my_regex, search_string_rc):
#             my_regex = r"([A|Y|N|9])\s" + country
#             vote_string = re.findall(my_regex, search_string_rc)
#             if len(vote_string) > 0:
#                 vote = str(vote_string[0])
#                 vote = vote.replace("Y", "YES")
#                 vote = vote.replace("N", "NO")
#                 vote = vote.replace("A", "ABSTAIN")
#                 vote = vote.replace("9", "INELIGIBLE")
#             else:
#                 vote = "ABSENT"
#             voting_record[country] = vote
#             # If there is no entry for the vote record, check if this might be a typo and record the resolution ID
#             if not vote_string:
#                 my_regex = r"\s([A-Z0-9]+)\s*" + country
#                 m = re.search(my_regex, search_string_rc)
#                 if m:
#                     suspect_string = m.group(1)
#                     countrynames_string = ''.join(countries)
#                     # Several suspects strings are due to absence (two country names without whitespace)
#                     countrynames_string = countrynames_string + "REPUBLICLEBANON"
#                     if suspect_string not in countrynames_string:
#                         possible_typos.append(suspect_string)
#                         output_typos = open(output_typos_filename, "a")
#                         print >> output_typos, str(res_number) + ", " + country + ": " + str(suspect_string)
#                         output_typos.close()
#         else:
#             voting_record[country] = "Missing"
#
#     # Adding entries that were missing due to typos, etc. in the original data source
#     # If date is missing, total voting membership is not extracted, so it has to be added manually as well
#     if res_number == "A/RES/55/209":
#         vote_date = "15.2.2001"
#         year = "2001"
#     if res_number == "A/RES/35/136":
#         vote_date = "11.12.1980"
#         year = "1980"
#     if res_number == "A/RES/32/105[M]":
#         vote_date = "14.12.1977"
#         year = "1977"
#         total = "149"
#     if res_number == "A/RES/66/230":
#         vote_date = "24.12.2011"
#         year = "2011"
#         total = "193"
#     if res_number == "A/RES/40/18":
#         title = "Bilateral nuclear-arms negotiations"
#     # Some information regarding country votes were incorrect due to typos
#     if res_number == "A/RES/33/91[F]":
#         voting_record["Iceland"] = "No"
#     if res_number == "A/RES/3228(XXIX)":
#         voting_record["Kuwait"] = "No"
#     if res_number == "A/RES/1668(XVI)":
#         voting_record["Tanganyika"] = "No"
#     if res_number == "A/RES/1668(XVI)":
#         voting_record["Romania"] = "No"
#
#         # Checking for missing values
#     if vote_date == "Missing":
#         missing_date_list.append(res_number)
#     if res_number == "Missing":
#         missing_res_number_list.append(data_sheet)
#     if yes_votes == "Missing":
#         missing_yes_vote_list.append(res_number)
#     if no_votes == "Missing":
#         missing_no_vote_list.append(res_number)
#     if abstentions == "Missing":
#         missing_abstentions_list.append(res_number)
#     if absent == "Missing":
#         missing_absent_list.append(res_number)
#     if total == "Missing":
#         missing_total_list.append(res_number)
#     if title == "Missing":
#         missing_title_list.append(res_number)
#     if not year.isdigit() or len(year) != 4:
#         missing_year_list.append(res_number)
#
#     ## Printing output to file
#     output_rc = open(ouput_rc_filename, "a")
#     print >> output_rc, str(
#         count) + ";" + data_sheet + ";" + res_number + ";" + "\"" + title + "\"" + ";" + "\"" + document + "\"" + ";" + meeting + ";" + vote_date + ";" + year + ";" + yes_votes + ";" + no_votes + ";" + abstentions + ";" + absent + ";" + total,
#     for country in sorted(countries):
#         print >> output_rc, ";" + voting_record[country],
#     print >> output_rc, "\n",
#     output_rc.close()
#     print
#     "Processed ID number: " + str(count)
#     count = count + 1
#
# if missing_res_number_list:
#     print
#     "\nData sheet number for entries with missing resolution ID:"
#     for element in missing_res_number_list:
#         print
#         element
# else:
#     print
#     "\nNo missing entries for resolution ID"
#
# if missing_title_list:
#     print
#     "\nData sheet number for entries with missing title:"
#     for element in missing_title_list:
#         print
#         element
# else:
#     print
#     "\nNo missing entries for title"
#
# if missing_countries:
#     print
#     "Country names (or other unknown strings):"
#     print
#     missing_countries
#     print
#     "See missing_3.txt for details. Typos for 'NY' have been manually fixed"
# else:
#     print
#     "No problems with missing country names (or other unknown strings in the roll call section)"
#
# if missing_date_list:
#     print
#     "\nResolution ID for entries with missing date:"
#     for element in missing_date_list:
#         print
#         element
# else:
#     print
#     "No missing entries for date"
#
# if missing_yes_vote_list:
#     print
#     "\nResolution ID for entries with missing yes-vote count:"
#     for element in missing_yes_vote_list:
#         print
#         element
# else:
#     print
#     "No missing entries for yes-vote count"
#
# if missing_no_vote_list:
#     print
#     "\nResolution ID for entries with missing no-vote count:"
#     for element in missing_no_vote_list:
#         print
#         element
# else:
#     print
#     "No missing entries for no-vote count"
#
# if missing_abstentions_list:
#     print
#     "\nResolution ID for entries with missing abstentions count:"
#     for element in missing_abstentions_list:
#         print
#         element
# else:
#     print
#     "No missing entries for abstentions count"
#
# if missing_absent_list:
#     print
#     "\nResolution ID for entries with missing absent count:"
#     for element in missing_absent_list:
#         print
#         element
# else:
#     print
#     "No missing entries for absent count"
#
# if missing_total_list:
#     print
#     "\nResolution ID for entries with missing total membership count:"
#     for element in missing_total_list:
#         print
#         element
# else:
#     print
#     "No missing entries for total membership count"
#
# if possible_typos:
#     print
#     "\nText of possible typos for vote record:"
#     for element in possible_typos:
#         print
#         element
# else:
#     print
#     "No suspect strings in roll call section found (some were manually fixed)"
#
# if missing_year_list:
#     print
#     "\nResolution ID for entries with missing (or apparently wrong) year:"
#     for element in missing_year_list:
#         print
#         element
# else:
#     print
#     "No missing/wrong entries for year"
