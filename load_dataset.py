#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from functools import reduce

AllOpioids = pd.read_excel(open('AllOpioids.xlsx', 'rb'), sheet_name='Expansion List', usecols='B', names=['description'], skiprows=12)
AllOpioids['label'] = 1
OpioidMeds = pd.read_excel(open('OpioidMeds.xlsx', 'rb'), sheet_name='Expansion List', usecols='B', names=['description'], skiprows=12)
OpioidMeds['label'] = 1

NonOpioidAnalgesics = pd.read_excel(open('NonOpioidAnalgesics.xlsx', 'rb'), sheet_name='Expansion List', usecols='B', names=['description'], skiprows=12)
NonOpioidAnalgesics['label'] = 0
NonOpioidAndNonAnalgesicPainMedications = pd.read_excel(open('NonOpioidAndNonAnalgesicPainMedications.xlsx', 'rb'), sheet_name='Expansion List', usecols='B', names=['description'], skiprows=12)
NonOpioidAndNonAnalgesicPainMedications['label'] = 0
NonOpioidAndNonAnalgesicPainMedications_2 = pd.read_excel(open('NonOpioidAndNonAnalgesicPainMedications-2.xlsx', 'rb'), sheet_name='Expansion List', usecols='B', names=['description'], skiprows=12)
NonOpioidAndNonAnalgesicPainMedications_2['label'] = 0
NonOpioidPainMedications = pd.read_excel(open('NonOpioidPainMedications.xlsx', 'rb'), sheet_name='Expansion List', usecols='B', names=['description'], skiprows=12)
NonOpioidPainMedications['label'] = 0

df = pd.concat([AllOpioids, OpioidMeds, NonOpioidAnalgesics, NonOpioidAndNonAnalgesicPainMedications,
            NonOpioidAndNonAnalgesicPainMedications_2, NonOpioidPainMedications], ignore_index=True).drop_duplicates(ignore_index=True)

# print(len(df[df['label'] == 0]))
# print(len(df[df['label'] == 1]))
# print(len(set(list(AllOpioids['description'])).intersection(set(list(OpioidMeds['description'])))))
# number of medications labeled as opioids: 1532
# number of medications labeled as non-opioids: 6541

# df.to_csv('dataset.csv')

# non_opioid = [NonOpioidAnalgesics, NonOpioidAndNonAnalgesicPainMedications, NonOpioidAndNonAnalgesicPainMedications_2, NonOpioidPainMedications]


# opioid_intersection = pd.merge(AllOpioids, OpioidMeds, how='inner', on=['description', 'label'])
# non_opioid_intersection = reduce(lambda left, right: pd.merge(left, right, on=['description', 'label'], 
#                                                               how='inner'), non_opioid)

# all_opioid = [AllOpioids, OpioidMeds,
#     NonOpioidAnalgesics, NonOpioidAndNonAnalgesicPainMedications, NonOpioidAndNonAnalgesicPainMedications_2, NonOpioidPainMedications]
# all_opioid_union = pd.merge(AllOpioids, OpioidMeds, how='outer', on=['description', 'label'])
# non_opioid_union = reduce(lambda left, right: pd.merge(left, right, on=['description', 'label'], 
#                                                               how='outer'), non_opioid)
# all_opioid_intersection = pd.merge(all_opioid_union, non_opioid_union, how='inner', on=['description'])

incorrect = pd.read_csv('temp.csv')
# for i in range(len(incorrect)):
#     desc = str(incorrect.loc[i]['0'])
#     print(desc)
#     if desc in list(NonOpioidAnalgesics['description']):
#         print('Non opioid analgesics')
#     if desc in list(NonOpioidAndNonAnalgesicPainMedications['description']):
#         print('Non opioid and non analgesic pain medications')
#     if desc in list(NonOpioidAndNonAnalgesicPainMedications_2['description']):
#         print('Non opioid pain medications (Grouping)')
#     if desc in list(NonOpioidPainMedications['description']):
#         print('Non opioid pain medications (Extensional)')

corrected_dataset = df.copy()

for i in range(len(incorrect)):
    desc = str(incorrect.loc[i]['0'])
    for j in range(len(df)):
        data = df.loc[j]
        if desc == data['description'] and data['label'] == 0:
            print(desc, data['label'])
            corrected_dataset.loc[j, 'label'] = 1
corrected_dataset.drop_duplicates()
corrected_dataset.to_csv('corrected_dataset.csv')
            
    
    
    