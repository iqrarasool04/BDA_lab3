import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
df = pd.read_excel(url)

# Data preprocessing
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

# Create a basket for each transaction
basket = (df[df['Country'] =="United Kingdom"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Encode the basket dataset
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

# Apply Apriori Algorithm
frequent_itemsets_apriori = apriori(basket_sets, min_support=0.03, use_colnames=True)

# Apply FP-Growth Algorithm
# frequent_itemsets_fpgrowth = fpgrowth(basket_sets, min_support=0.03, use_colnames=True)

# Analyze the results
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.5)
# rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=0.5)

# Interpretation and Insights
print("Apriori Rules:")
print(rules_apriori.head())
# print("\nFP-Growth Rules:")
# print(rules_fpgrowth.head())

# Comparison and Evaluation
# You can compare the performance and outcomes of the two algorithms here
