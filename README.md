# kritis_basket
My python code implements market basket analysis using the apyori package on data that is not in the form required by the apriori function to work on. Apriori takes the input in the form of list of lists with the inner lists being the items bought in the same transaction. So far, the tutorials that I have seen online use a data frame that already has each  row representing a transaction having the respective items. 
However, my data frame has all the items in a single column called 'model_id' meaning each row was added when a purchase was made. Thus, I did'nt have a ready made transaction table and I needed to create it. Also, the apriori's rules in Python are not easy to view. Hence, my code has the steps before and after using apriori to make it easier to use it in Python.
