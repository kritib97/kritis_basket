# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:04:12 2019

@author: Kriti.Biswas
"""
import numpy as np
import pandas as pd
from apyori import apriori

z3 = pd.read_csv('mba.csv')

# Create the transaction column
z3['tran'] = z3['Customer_ID'].astype(str)  + z3['Date_added'].astype(str)

v1 = z3[['model_id', 'tran']]
v1.model_id = v1['model_id'].astype(str) # Convert column model id to string type

a = v1.groupby('tran')['model_id'].apply(list) # get the model IDs of items bought in each transaction
a = pd.DataFrame(a)
a.reset_index(inplace=True)
del a['tran']

# Create a list of item lists
from itertools import chain
transactions = []
for i in range(0,len(a)): 
    lst = a.loc[i,:]
    res = list(chain.from_iterable(i if isinstance(i, list) else [i] for i in lst))
    transactions.append(res)
    
from apyori import apriori
rules = apriori(transactions, min_support = 0.001, min_confidence = 0.5)
results = list(rules)
print(results[0])
print(len(results))

    
# Viewing the rules in a dataframe
r = pd.DataFrame(columns=['LHS','RHS', 'Support', 'Confidence', 'Lift'])

for item in results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    lhs=[]
    if len(items)>2:
        for i in range(0,len(items)-1):
           lhs.append(items[i])
        rhs=items[len(items)-1]
    else:
        lhs.append(items[0])
        rhs=items[1]

    #second index of the inner list
    Support = item[1]

    #third index of the list located at 0th
    #of the third index of the inner list
    Confidence=item[2][0][2]
    Lift=item[2][0][3]
    
    # Add these values to a dataframe
    newrow = [lhs, rhs, Support, Confidence, Lift]
    r = r.append(pd.DataFrame(columns=r.columns, data=[newrow]), ignore_index=True)    

# Format the dataframe
pd.options.display.float_format = '{:,.5f}'.format
r = r.astype({"Support": float, "Confidence": float, "Lift": float})
