### 1 Introduction

In this project, I’m planning to analyze and explore the Census data set. I found a data set on UCI Machine Learning Repository, which is known as "Census Income" dataset. This is a dataset from 1996 and there are 48842 instances in it. Although it is a rather outdated data set, I still think some meaningful and interesting conclusions can be obtained from this project as it’s closely related to sociology.

### **2 Problem**

Given a series of a person’s basic information and the system should predict has an annual income over 50k dollars.

### **3 Input**

The input is a series of basic information about a person being analyzed, including age, workclass, fnlwgt and other 11 attributes.

**age**: continuous. 
 **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
 **fnlwgt**: continuous. 
 **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
 **education**-**num**: continuous. 
 **marital**-**status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
 **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
 **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
 **race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
 **sex**: Female, Male. 
 **capital**-**gain**: continuous. 
 **capital**-**loss**: continuous. 
 **hours**-**per**-**week**: continuous. 
 **native**-**country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

### **4 Output** 

The output is the prediction of whether the person being analyzed is earning more than 50k/year or not (True or False).

### **5 ML Technique**

I decide to implement the Naiive Bayes classifier as it’s one of the fastest and most accurate classifier. There are some previous works on this data set. For example, nearest-neighbor (1) algorithm has been implemented on this data set and gives an error result of 21.42. If time permits, I’m also planning to develop GUI so that we can have better insight to the data set. 

### **6 Dataset**

URL: <http://archive.ics.uci.edu/ml/datasets/Adult>

This data was extracted from the census bureau database found at http://www.census.gov/ftp/pub/DES/www/welcome.html