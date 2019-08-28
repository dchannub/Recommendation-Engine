# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from efficient_apriori import apriori as ef_apr
from mlxtend.frequent_patterns import apriori
from apyori import apriori as apy_apr
from mlxtend.frequent_patterns import association_rules
import csv


"""
Variable          Description

CustomerID        Unique ID for customer
InvoiceNo         Invoice number of transaction
Quantity          Quantity of item bought
InvoiceDate       Invoice Date
UnitPrice         Unit Price for item
Country           Country Code
StockCode         Item ID

"""

#warnings.filterwarnings('ignore')

os.chdir("C:/Users/dchannubhotl.EAD/Desktop/Data Science Capability program/POC")
df_1 = pd.read_csv('train_5UKooLv.csv')
df_test_1 = pd.read_csv('test_J1hm2KQ.csv')

df= pd.concat([df_1,df_test_1])

df = df.reset_index(drop=True)

df = df.query('Quantity > 0 & CustomerID != 0')
df.head()

grouped_0 = df.groupby(['CustomerID'])
grouped = grouped_0.aggregate(lambda x: list(x))

trans = grouped['StockCode'].tolist()
trans[2]
grouped.head()


#temsets, rules = apriori(trans, min_support=0.15,min_confidence=.95)
#itemsets, rules = apriori(trans, min_support=0.14,min_confidence=.80)
itemsets_20, rules_20 = apriori(trans, min_support=0.19,min_confidence=.50)
#mostFrequent_itemsets, MostFrequent_rules = apriori(trans, min_support=0.38,min_confidence=.85)


x1 = itemsets_20[1]

ass_rules = apy_apr(trans,min_support=.38,min_confidence=0.7,min_lift=1.2)
ass_results =list(ass_rules)
y=ass_results[3]
x =ass_results[3][2][0][0]
x =str(ass_results[3][2][0][1])

grouped_mlxtnd = (df.groupby['CustomerID'].sum().unstack().reset_index().fillna(0).set_index('CustomerID'))

frquent_itemsets = apriori(grouped_0,min_support=0.7)
mlxtnd_rules = association_rules(itemsets_20,metric="lift",min_threshold=1)


print(str(rules_20[10000]))
# Most Frequently bought items in the whole invoice.


# Item frequncy plot

df_test_1 = pd.read_csv('test_J1hm2KQ.csv')
df_test = df_test_1.query('Quantity > 0')
df_test_negative = df_test_1.query('Quantity <= 0')


# handling the negative data

grouped_test_neg = df_test_negative.groupby(['CustomerID'])
grouped_test_neg = grouped_test_neg.aggregate(lambda x: list(x))

test_df_neg = grouped_test_neg[['InvoiceNo','StockCode']]


dict_exp_items_neg = test_df_neg.to_dict()
Invoice_Stocks_neg = dict_exp_items_neg['StockCode']

# making all the values negative = 0

Invoice_Stocks_neg_sam = {}



grouped_test = df_test_1.groupby(['CustomerID'])
#grouped_test = df_test.groupby(['CustomerID'])
grouped_test = grouped_test.aggregate(lambda x: list(x))

test_df = grouped_test[['InvoiceNo','StockCode']]


dict_exp_items = test_df.to_dict()
Invoice_Stocks = dict_exp_items['StockCode']



for ky,vl in Invoice_Stocks_neg.items():
    if ky not in Invoice_Stocks.keys():
        Invoice_Stocks[ky] = ""



Recommendation_dict = dict()
Recommended_list = []
existing_list = ""
has_recomendation = "No"
most_popular_recommeded_list = []
invoice_has_reommendation = "No"
#recommended_item = ""

Invoice_Stocks[339156]

for invoice,items in Invoice_Stocks.items():
    Recommended_list = [] 
    recommended_item = []
    #print(row)
    #str_items = str(items)
    #str_items = str_items.replace(" ","")
    #str_items = str_items.replace("['","")
    #str_items = str_items.replace("']","")
    #str_items = str_items.replace("'","")
    #items_list = str_items.split(",")
    has_recomendation = "No"
    invoice_has_reommendation = "No"                             
    #for each_item in items_list:       
    for i, rule in enumerate(rules_20):
        #print(str(rule))
        str_items = str(rule)        
        str_items = str_items.split("->")
        str_items = str_items[0]
        str_items = str_items.replace(" ","")
        str_items = str_items.replace("{","")
        str_items = str_items.replace("}","")
        items_list = str_items.split(",") 
        #print("Item List"+str(items_list))
        #print("list lenght"+str(len(items_list)))
        #rule_list =   items_list.join(",") 
        for lhs_item in items_list:
            #print("again single"+str(lhs_item)) 
            if (lhs_item in items):
                has_recomendation = "Yes"
            
        for t_rhs in rule.rhs:
            if (has_recomendation == "Yes" and rule.rhs not in Recommended_list and str(t_rhs) not in items):# and len(Recommended_list) <= round(.35*(len(items)))):
                #print("Has Recommendation :"+has_recomendation+"- for invoice"+str(invoice))
                #print("recommended list = "+str(Recommended_list))
                Recommended_list.append(rule.rhs)
        #if len(Recommended_list) >= round(.35*(len(items))):
            #break
    Recommendation_dict[invoice]=Recommended_list
    #Recommended_list = []
    #print("Has Recommendation :"+has_recomendation+"- for invoice"+str(invoice))#+": Recommended Items = "+str(Recommended_list))
           
    if (has_recomendation != "Yes"):
        print("Has Recommendation :"+has_recomendation+"- for invoice"+str(invoice))#+": Recommended Items = "+str(Recommended_list))
        Recommended_list = []
        most_popular_recommeded_list = []
        invoice_has_reommendation = "No"
        has_recomendation = "No"           
        for most_rules in MostFrequent_rules:
            most_rules_str = str(most_rules.rhs)
            most_rules_str = most_rules_str.replace(" ","")
            most_rules_str = most_rules_str.replace("(","")
            most_rules_str = most_rules_str.replace(",","")
            most_rules_str = most_rules_str.replace("'","")
            most_rules_str = most_rules_str.replace(")","")
                       
            
            for rhs in rule.rhs:                
                for lhs in most_rules.rhs:            
                    #print(str(lhs))
                    #print(str(rhs))            
                    if (str(lhs) not in str(most_popular_recommeded_list)) and str(rhs) not in str(items):# and len(Recommended_list) <= round(.35*(len(items)))): 
                        print("invoice = "+str(invoice))
                        print("Recomendation: " +has_recomendation)
                        print("Recommended Items"+str(Recommended_list))
                        Recommended_list.append(most_rules.rhs)
                        Recommendation_dict[invoice]=Recommended_list
                        print("No Recommendation for invoice"+str(invoice)+" , Item "+str(rule))
                        print("recommended list = "+str(Recommended_list))
                        print("most popular list:"+str(most_popular_recommeded_list)+": rhs : "+str(most_rules.rhs))
                        most_popular_recommeded_list = most_rules.rhs
                
    #print("Current Invoice"+str(invoice))#+": Recommended Items = "+str(Recommended_list))
    
                
    
# Export solution to a CSV File to share with the customer.

loc=1
#cnames = ['CustomerID','RecommendedStockID']
solution_df = pd.DataFrame(columns= ("CustomerID","Items"))
                         
for invoice,items in Recommendation_dict.items():
    str_itms = str(items)
    str_items = str_itms.replace("(","")
    str_items = str_items.replace(")","")
    str_items = str_items.replace(" ","")
    str_items = str_items.replace(",,",",")
    str_items = str_items.replace(",]","]")
    solution_df.loc[loc]=[invoice,str_items]
    loc=loc+1
                           
solution_df.to_csv(r'result_23_July_2019_Office.csv',index=False)




def compute_CK(LK_, k):
	CK = []
	for i in range(len(LK_)):
		for j in range(i+1, len(LK_)): # enumerate all combinations in the Lk-1 itemsets
			L1 = sorted(list(LK_[i]))[:k-2]
			L2 = sorted(list(LK_[j]))[:k-2]
			if L1 == L2: # if the first k-1 terms are the same in two itemsets, merge the two itemsets
				new_candidate = frozenset(sorted(list(LK_[i] | LK_[j]))) # set union
				CK.append(new_candidate) 
	return sorted(CK)


def compute_LK(D, CK, num_trans, min_support):
	support_count = {}
	for item in D: # traverse through the data set
		for candidate in CK: # traverse through the candidate list
			if candidate.issubset(item): # check if each of the candidate is a subset of each item
				if not candidate in support_count:
					support_count[candidate] = 1
				else: support_count[candidate] += 1
	LK = []
	supportK = {}
	for candidate, count in sorted(support_count.items(), key=lambda x: x[0]):
		support = count / num_trans
		if support >= min_support:
			LK.append(candidate)
			supportK[candidate] = count
	return sorted(LK), supportK

def compute_C1_and_L1_itemset(data, num_trans, min_support):
	#---compute C1---#
	C1 = {}
	for transaction in data:
		for item in transaction:
			if not item in C1:
				C1[item] = 1
			else: C1[item] += 1
	#---compute L1---#
	L1 = []
	support1 = {}
	for candidate, count in sorted(C1.items(), key=lambda x: x[0]):
		support = count / num_trans
		if support >= min_support:
			L1.insert(0, [candidate])
			support1[frozenset([candidate])] = count
	return list(map(frozenset, sorted(L1))), support1, C1


def my_apriori(data, min_support):
	D = sorted(list(map(set, data)))
	num_trans = float(len(D))
	L1, support_list, CK = compute_C1_and_L1_itemset(data, num_trans, min_support)
	L = [L1]
	k = 1

	while (True): # create superset k until the k-th set is empty (ie, len == 0)
		print('Running Apriori: the %i-th iteration with %i candidates...' % (k, len(CK)))
		k += 1      
		CK = compute_CK(LK_=L[-1], k=k)     
		LK, supportK = compute_LK(D, CK, num_trans, min_support)
		if len(LK) == 0: 
			L = [sorted([tuple(sorted(list(itemset), key=lambda x: int(x))) for itemset in LK]) for LK in L]
			support_list = dict((tuple(sorted(list(k), key=lambda x: int(x))), v) for k, v in support_list.items())
			print('Running Apriori: the %i-th iteration. Terminating ...' % (k-1))
			break
		else:
			L.append(LK)
			support_list.update(supportK)
	return L, support_list

D = sorted(list(map(set, trans)))
num_trans = float(len(D))
  

it,su = my_apriori(trans,min_support=.30)


LK, supportK = compute_LK(D, CK, num_trans, min_support)


L1, support_list, CK = compute_C1_and_L1_itemset(trans, num_trans, 0.3)
CK_CK = dict(CK)

compute_LK(trans,CK_CK,num_trans,0.3)

C1 = {}
for transaction in trans:
		for item in transaction:
			if not item in C1:
				C1[item] = 1
			else: C1[item] += 1
            
            
            
L1 = []
support1 = {}
for candidate, count in sorted(C1.items(), key=lambda x: x[0]):
	support = count / num_trans
	if support >= .3:
		L1.insert(0, [candidate])
		support1[frozenset([candidate])] = count