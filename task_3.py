import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#preparing dataset
dataset = [['I1','I2','I5'], ['I2','I4'], ['I1','I3'], ['I5','I4','I3'], ['I2']]

#encoding dataset
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns = te.columns_)

#applying apriori
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

#calculating length of itemsets
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x:len(x))
filtered_itemsets = frequent_itemsets[(frequent_itemsets['length']>=1) & (frequent_itemsets['support']>=0.2)]

#extracting association rules
rules = association_rules(filtered_itemsets, metric="confidence", min_threshold=0.7, support_only=False)
print(rules)
