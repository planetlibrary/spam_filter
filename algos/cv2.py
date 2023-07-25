from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import math

class NBLogCount:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        dfdic = {'text' : self.X, 'label':self.y}
        df = pd.DataFrame(dfdic)
        
        print(df)
        
        df1= df[df['label'] == 1]
        df2 = df[df['label'] == 0]

        cv1 = CountVectorizer()
        cv2 = CountVectorizer()

        self.p_class1 = len(df1.label)/len(df.label)
        self.p_class2 = len(df2.label)/len(df.label)

        xcount1 = cv1.fit_transform(df1.text) 
        xcount2 = cv2.fit_transform(df2.text) 

        countlist1 = xcount1.toarray().sum(axis = 0)
        wordlist1 = cv1.get_feature_names_out()
        self.feature_dic1  = dict(zip(wordlist1, countlist1))

        countlist2 = xcount2.toarray().sum(axis = 0)
        wordlist2 = cv2.get_feature_names_out()
        self.feature_dic2  = dict(zip(wordlist2, countlist2))

        self.N1 = xcount1.toarray().sum()
        self.N2 = xcount2.toarray().sum()
          
    def fit(self, word, min_support = 0.01):
        self.alpha = min_support
        try:
            numerator = (self.feature_dic1[word]/self.N1)*self.p_class1
        except:
            numerator=self.alpha #not in class 1
        try:
            denominator = (self.feature_dic2[word]/self.N2)*self.p_class2
        except:
            denominator=self.alpha #not in class 2

        val = numerator/denominator
        log_val = abs(math.log(val))
        return math.exp(log_val) - 1

               
# X = [ ['This ','is',' a',' dog'],['This',' is',' a',' cat'],['That',' is',' a',' bitch'],['That',' is',' a',' pussy'],['dog']]
# y = [1,1,0,0,0]

# X = np.array(X)
# y = np.array(y)

# nbl = NBLogCount (X,y)

# words = ['dog','cat','pussy', 'this']
# for word in words:
#     print('The weight of ',word, 'is: ', nbl.fit(word))