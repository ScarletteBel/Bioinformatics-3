# ChEMBL Database is the Database that will be used, containinig curated bioactivity data of more than 2 million compounds.
# it is compiled from more than 76,000 documents, 1.2 million assays and the data spans 13,000 targets and 1,800 cells and 33,000 indications.

import pandas as pd 
from chembl_webresource_client.new_client import new_client

#Serch for the target protein 

target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)
print(targets)
