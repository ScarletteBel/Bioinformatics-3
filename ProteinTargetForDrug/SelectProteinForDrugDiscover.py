# Final Project   Scarlette Bello       c0860234

#The data used in this project is biological data, the analysis of this kind of data using computational tools is also know as bioinformatics.
#Machine learnijng models applied in bioinformatics are also known as "quantitive structure activity relationship" and is applied for drug discover efforts.
#Machine learning models  in this bioinformatics area allows to understand the origins of the bioactivity (how chemical components interact with microorganisms and proteins).
#In less words, this analizis help on the drug design improvement.

#The dataset used in this project is calles 'Coronavirus', downloaded form CHEMBL Database.
# ChEMBL Database contains curated bioactivity data.

#The data base has 7 targets. Targets make reference to te Target organism or Target protein that the drug would act on (it can activate or inhibit the biological activity).

import pandas as pd 
from chembl_webresource_client.new_client import new_client
import matplotlib.pyplot as plt

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski



#Serch for the target protein 

target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)
print(targets)

target_selected = targets.target_chembl_id[4]
print(target_selected)



activity = new_client.activity
res = activity.filter(target_chembl_id=target_selected).filter(standard_type='IC50')

bioactivity_df = pd.DataFrame.from_dict(res)
print(bioactivity_df)



#Data preprocessing of the bioactivity data...


bioactivity_class = []
for i in bioactivity_df.standard_value:
  if float(i) >= 10000:
    bioactivity_class.append("inactive")
  elif float(i) <= 1000:
    bioactivity_class.append("active")
  else:
    bioactivity_class.append("active")

mol_cid = []
for i in bioactivity_df.molecule_chembl_id:
  mol_cid.append(i)


canonical_smiles = []
for i in bioactivity_df.canonical_smiles:
  canonical_smiles.append(i)
    

standard_value = []
for i in bioactivity_df.standard_value:
  standard_value.append(i)


data_tuples = list(zip(mol_cid, canonical_smiles, bioactivity_class, standard_value))
df3 = pd.DataFrame( data_tuples,  columns=['molecule_chembl_id', 'canonical_smiles', 'bioactivity_class', 'standard_value'])
print(df3)

bio_df3 = pd.get_dummies(df3, columns=['bioactivity_class'])
bio_df3 = pd.DataFrame.drop(bio_df3, columns = ['bioactivity_class_inactive'])
print(bio_df3)





df_no_smiles = bio_df3.drop(columns='canonical_smiles')

smiles = []

for i in bio_df3.canonical_smiles.tolist():
  cpd = str(i).split('.')
  cpd_longest = max(cpd, key = len)
  smiles.append(cpd_longest)

smiles = pd.Series(smiles, name = 'canonical_smiles')

df_clean_smiles = pd.concat([df_no_smiles,smiles], axis=1)


# Inspired by: https://codeocean.com/explore/capsules?query=tag:data-curation

def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors


df_lipinski = lipinski(df_clean_smiles.canonical_smiles)
print(df_lipinski)


df_combined = pd.concat([bio_df3,df_lipinski], axis=1)
print(df_combined)

# https://github.com/chaninlab/estrogen-receptor-alpha-qsar/blob/master/02_ER_alpha_RO5.ipynb

import numpy as np

def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)
        
    return x


print(df_combined.standard_value.describe())


# #Data visualization...

# plt.scatter(x =df_combined['LogP'], y=df_combined['bioactivity_class'])
# plt.show()










