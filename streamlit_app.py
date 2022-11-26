import streamlit as st
import pandas as pd
import sys
import subprocess
import math
import re
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',package])
import_or_install("faker")
import_or_install("PIL")
from PIL import Image
import_or_install("requests")
import requests
#r= requests.get("https://upload.wikimedia.org/wikipedia/en/thumb/7/79/University_of_Chicago_shield.svg/195px-University_of_Chicago_shield.svg.png", stream=True)
im = Image.open('Logo1.png')

from faker import Faker
import_or_install("seaborn")
import seaborn as sns
import_or_install("recordlinkage")
import recordlinkage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from recordlinkage.datasets import load_febrl4
from recordlinkage.preprocessing import phonetic
from recordlinkage import Compare
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import_or_install("snorkel")
import snorkel
st.set_page_config(
    page_title="Record_Linkage",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded"
)
import random as rn

header=st.container()
dataset=st.container()
featurest=st.container()
model_training=st.container()
faker_data=st.container()
upload_data=st.container()
encryption_required=st.container()
encrypion_not_required=st.container()
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel

st.set_option('deprecation.showPyplotGlobalUse', False)
    

st.markdown(
   """ <style> 
   .font2 {
font-family:"serif";
    font-size: 160%;
    color:#000000;    
    line-height: 100%;
    background-color: #FBCEB1;
    padding: 0.4em;
    letter-spacing: -0.05em;
    word-break: normal;
    border-radius: 10px;
    } 
      .font11 {
font-family:"serif";
    font-size: 250%;
    color:#FFFFFF;    
    line-height: 100%;
    background-color: #06038D;
    padding: 0.4em;
    letter-spacing: -0.05em;
    word-break: normal;
    border-radius: 10px;
    } 
    .font1 {
font-family:"serif";
    font-size: 160%;
    color:#00FFFF;    
    line-height: 100%;
    background-color: #36454F;
    padding: 0.4em;
    letter-spacing: -0.05em;
    word-break: normal;
    border-radius: 5px;
}
        .font3 {
font-family:"serif";
    font-size: 100%;
    color:#000000;    
    font-weight: 300;
    line-height: 100%;
    background-color: #FFFFFF;
    padding: 0.4em;
    letter-spacing: -0.05em;
    word-break: normal;
    border-radius: 5px;
    /font-style: italic;
}
</style> 

""",
        unsafe_allow_html=True,
    )
NOMATCH = 0
UNKNOWN = -1
MATCH = 1
@labeling_function()
def name(x):
    return MATCH if  x.given_name>0.9 and x.surname>0.9 else UNKNOWN
@labeling_function()
def name_date(x):
    return MATCH if  x.given_name>0.9 and x.surname>0.9 and x.date_of_birth>0.7 else UNKNOWN
@labeling_function()
def name_rev(x):
    return NOMATCH if  x.given_name<0.6 or x.surname<0.6 else UNKNOWN    
@labeling_function()
def address(x):
    return MATCH if  x.address>0.6 else UNKNOWN
@labeling_function()
def address_rev(x):
    return NOMATCH if  x.address<0.2 else UNKNOWN   
lfs = [name,address, name_rev, address_rev,name_date]
applier = PandasLFApplier(lfs=lfs)
def plot_metrics(model,metrics_list,x_test,y_test):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
        
def candidate_links_func(dfA,dfB,blocker): 
      indexer = recordlinkage.Index()
      if blocker!="":
        indexer.block(blocker)
        candidate_links = indexer.index(dfA, dfB)
      else:
        a=list(dfA.index)
        b=list(dfB.index)
        candidate_links=pd.MultiIndex.from_product([a,b]) 
      return candidate_links  

def data1(dfA,dfB,blocker=""):
      candidate_links=candidate_links_func(dfA,dfB,blocker)
      compare = Compare()
      compare.string('First Name', 'First Name', method='cosine', label="First Name")
      compare.string('Last Name', 'Last Name', method='cosine', label="Last Name")
      compare.string('Suburb', 'Suburb', method='cosine', label="Suburb")
      compare.string('State', 'State', method='cosine', label="State")
      compare.string('Address', 'Address', method='cosine', label="Address")
      compare.string("Date of Birth","Date of Birth",method='cosine', label="Date of Birth")
      features = compare.compute(candidate_links, dfA, dfB)
      return features

fake = Faker(42)   
def data_creation(entries):
    given_name = []
    surname = []
    street_number=[]
    address_1=[]
    address_2=[]
    suburb=[]
    state = []
    postcode = []
    date_of_birth = []
    soc_sec_id = []
    
    for q in range(entries):
        given_name.append(fake.first_name())
        surname.append(fake.last_name())
        street_number.append(fake.building_number())
        address_1.append(fake.street_suffix())
        address_2.append(fake.street_name())
        suburb.append(fake.city())
        state.append(fake.state())
        postcode.append(fake.zipcode())
        soc_sec_id.append(fake.ssn())
        date_of_birth.append(fake.date_of_birth())
        
    df = pd.DataFrame(list(zip(given_name, surname, street_number, address_1, address_2, suburb,  postcode,state,date_of_birth,soc_sec_id)), 
                      columns= ['given_name', 'surname', 'street_number', 'address_1', 'address_2', 'suburb','postcode', 'state','date_of_birth','soc_sec_id'])
    return df  
class_names = ["No Match", "Match"]
my_dict= {'name': 'First Name', 'givenname': 'First Name', 'fname': 'First Name', 'firstname': 'First Name',
          'surname': 'Last Name', 'familyname': 'Last Name', 'lname': 'Last Name', 'lastname': 'Last Name',
          'streetno': 'Street Number', 'stno': 'Street Number', 'streetnumber': 'Street Number',
          'streetaddress': 'Address', 'addr': 'Address', 'addressline1': 'Address1', 'address':'Address', 'address1':'Address1',
          'unitnumber': 'Address2', 'apartmentnumber': 'Address', 'addr2': 'Address2', 'addressline2': 'Address2', 'address2': 'Address2',
          'county': 'Suburb', 'city': 'Suburb', 'area': 'Suburb', 'region': 'Suburb', 'suburb':'Suburb',
          'zipcode':'Postcode', 'areacode':'Postcode','zip':'Postcode', 'postalcode':'Postcode', 'postcode':'Postcode',
          'state': 'State',
          'dob':'Date of Birth', 'birthdate':'Date of Birth', 'dateofbirthddmmyy':'Date of Birth', 'dateofbirthmmddyy':'Date of Birth', 'dateofbirthddmmyyyy':'Date of Birth', 'dateofbirthmmddyyyy':'Date of Birth', 'dobddmmyy':'Date of Birth', 'dobmmddyy':'Date of Birth', 'dobddmmyyyy':'Date of Birth', 'dobmmddyyyy':'Date of Birth', 'dateofbirth':'Date of Birth',
          'ssn':'Social Security Number', 'socsecid': 'Social Security Number', 'socialsecuritynumber':'Social Security Number', 'ssa':'Social Security Number', 'socialsecuritycard':'Social Security Number', 'ssid':'Social Security Number', 'socialsecuritynumer':'Social Security Number',
          'contactnumber':'Phone Number', 'number':'Phone Number', 'phone':'Phone Number', 'phno':'Phone Number', 'phoneo':'Phone Number', 'phnumber':'Phone Number', 'mobile':'Phone Number', 'mobileno':'Phone Number', 'mobilenumber':'Phone Number', 'cellphone':'Phone Number', 'cellphoneno':'Phone Number', 'cellphonenumber':'Phone Number', 'phonenumber':'Phone Number',
          'email':'Email Address', 'emailid':'Email Address', 'emailaddress':'Email Address'}
st.sidebar.title("Interactive") 


models=st.sidebar.selectbox("How would you Feature Importance to be modeled?",("Gradient Boosting", "Logistic Regression", "Weak Supervision"))

metrics = st.sidebar.multiselect("What metrics to plot for the model?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))


def column_matching(column_names):
    canonical_lst=[]
    new_column_names=[]
    def standard_name(col_name):
        col_name= ''.join(col_name.split()).lower()
        col_name= re.sub("[^A-Za-z0-9]", '', col_name)
        if col_name in my_dict:
            col_name= my_dict[col_name]
        else:
            canonical_lst.append(col_name)
        new_column_names.append(col_name)
    for col in  column_names:
        standard_name(col)
    return new_column_names,canonical_lst

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def data():
    dfA, dfB, true_links = load_febrl4(return_links=True)
    new_column_names,canonical_lst=column_matching(list(dfA.columns))
    dfA.columns=new_column_names
    new_column_names,canonical_lst=column_matching(list(dfB.columns))
    dfB.columns=new_column_names
    dfA["initials"] = (dfA["First Name"].str[0]  + dfA["Last Name"].str[0])
    dfB["initials"] = (dfB["First Name"].str[0]  + dfB["Last Name"].str[0])
    dfA["Date of Birth"] = dfA["Date of Birth"].str.replace('-', "")
    dfB["Date of Birth"] = dfB["Date of Birth"].str.replace('-', "")
    dfA["soc_sec_id"] = dfA["Social Security Number"].str.replace('-', "")
    dfB["soc_sec_id"] = dfB["Social Security Number"].str.replace('-', "")
    dfA['Address']=dfA['Street Number']+" "+dfA['Address1']+" "+dfA['Address2']
    dfB['Address']=dfB['Street Number']+" "+dfB['Address1']+" "+dfB['Address2'] 
    features=data1(dfA,dfB,"initials")
    return dfA,dfB,true_links,features


with header:
    #r= requests.get("https://miro.medium.com/max/1400/1*VSMuHlqP5FFzhbymr0gOSQ.png", stream=True)
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open('Picture1.png')
        st.image(image, width=380)
    with col2:
        st.write("Record linkage (also known as data matching, entity resolution, and many other terms) is the task of finding records in a data set that refer to the same entity across different data sources. Record linkage is necessary when joining different data sets based on entities that may or may not share a common identifier, which may be due to differences in record shape, storage location, or curator style or preference.")
    st.markdown("""# Capstone Project""")
    st.markdown('<p class="font11">Welcome to Record Linkage</p>',unsafe_allow_html=True)
    
with dataset:
    st.markdown('<p class="font2">Data from Record Linkage Package used for Feature Selection Modeling</p>', unsafe_allow_html=True)
    dfA, dfB, true_links,features=data()
    st.markdown('<p class="font3">Few Lines of Data', unsafe_allow_html=True)
    st.write(dfA.head(5))  

with  featurest:
    st.markdown('<p class="font2">Modelling Features</p>', unsafe_allow_html=True)
    features['Target']=features.index.isin(true_links)
    features['Target']=features['Target'].astype(int)
    data=features.reset_index(drop=True)
    X=data.drop(['Target'],axis=1)
    Y=data['Target']
    X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2)
    st.markdown('<p class="font3">Input train Dataset: Applied cosine similarity between two records', unsafe_allow_html=True)
    st.write(X_train[:5])
    st.markdown('<p class="font3">Actual Output train Dataset', unsafe_allow_html=True)
    st.write(y_train[:5])
    y_train_dst=y_train.value_counts()
    st.markdown('<p class="font3">Matches and Non Matches Distribution in Train Data', unsafe_allow_html=True)
    st.write(y_train_dst)

@st.cache(suppress_st_warning=True)
def Gradient(n_estimators,max_depth):
        global metrics
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.1,max_depth=max_depth, random_state=42).fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        model_final=clf
        y_pred=model_final.predict(X_test)
        accuracy = model_final.score(X_test, y_test)
        y_pred = model_final.predict(X_test)
        return accuracy,model_final,X_test,y_test,y_pred
@st.cache(suppress_st_warning=True)
def Logistic(penalty,C):
        global metrics
        lr=LogisticRegression(class_weight="balanced",penalty=penalty,C=C,solver="saga", l1_ratio=0.5)
        lr.fit(X_train,y_train)
        model_final=lr
        y_pred=model_final.predict(X_test)
        accuracy = model_final.score(X_test, y_test)
        y_pred = model_final.predict(X_test)
        return accuracy,model_final,X_test,y_test,y_pred
@st.cache(suppress_st_warning=True)
def WS(X_train,X_test,model_type):
       global metrics
       L_train = applier.apply(df=X_train)
       L_test = applier.apply(df=X_test)
       if model_type=="MajorityLabelVoter":
           model_final = MajorityLabelVoter()
           accuracy = model_final.score(L=L_test, Y=X_test.Target, tie_break_policy="random")["accuracy"] 
       else:
           model_final = LabelModel(cardinality=3, verbose=True)
           model_final.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)
           accuracy = model_final.score(L=L_test, Y=X_test.Target, tie_break_policy="abstain")["accuracy"]
       sol=model_final.predict(L_test)
       return  model_final,sol,y_test,accuracy,L_train
    
with  model_training:
    if models=="Gradient Boosting":
        st.markdown('<p class="font2">Applying Gradient Boosting to Model Feature Importance</p>', unsafe_allow_html=True)
        n_estimators=st.slider("What would be the number of estimators of the model?", min_value=10,max_value=100,value=100,step=10)
        max_depth=st.slider("What would be the max_depth of the model?", min_value=1,max_value=10,value=8,step=1)
        accuracy,model_final,X_test,y_test,y_pred=Gradient(n_estimators,max_depth)
        plot1=pd.DataFrame()
        plot1['Features']=list(X_train.columns)
        plot1['Importance']=model_final.feature_importances_
        plot1 = plot1.set_index('Features')
        st.bar_chart(plot1)
        st.markdown('<p class="font2">Performance of Model on Test Data', unsafe_allow_html=True)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
        plot_metrics(model_final,metrics,X_test,y_test)
    elif models=="Logistic Regression":
        st.markdown('<p class="font2">Applying Logistic Regression to Model Feature Importance</p>', unsafe_allow_html=True)
    
        penalty=st.select_slider('Select penalty type',options=['l1', 'l2', 'elasticnet', 'none'],value=('l2'))
        C=st.slider("What would be the value of C(Inverse of regularization strength)?", min_value=0.1,max_value=2.0,value=1.0,step=0.1)        
        accuracy,model_final,X_test,y_test,y_pred=Logistic(penalty,C)
        plot1=pd.DataFrame()
        plot1['Features']=list(X_train.columns)
        plot1['Importance']=model_final.coef_[0]
        plot1 = plot1.set_index('Features')
        st.bar_chart(plot1)
        st.markdown('<p class="font3">Performance of Model on Test Data', unsafe_allow_html=True)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
        plot_metrics(model_final,metrics,X_test,y_test)
    elif models=="Weak Supervision":
       st.markdown('<p class="font2">Applying Snorkel to Model Feature Importance</p>', unsafe_allow_html=True)
       model_type=st.select_slider('Select model type',options=['MajorityLabelVoter', 'LabelModel'],value=('LabelModel'))
       data=features
       df = data.sample(frac=1)
       X_train=df.head(int(len(df)*0.8))
       X_test=df.tail(int(len(df)*0.2)) 
       model_final,sol,y_test,accuracy,L_train =WS(X_train,X_test,model_type)
       out1=LFAnalysis(L=L_train, lfs=lfs).lf_summary()
       out1=out1[['Coverage','Overlaps','Conflicts']]
       st.write(out1)
       st.write("Accuracy: ", accuracy.round(2))
       X_test1=X_test.copy()
       X_test1['CLASS']=sol
       X_test1=X_test1.loc[(X_test1.CLASS==0) | (X_test1.CLASS==1)]
       if "Confusion Matrix" in metrics:
           st.subheader("Confusion Matrix")
           cm=confusion_matrix(X_test1.Target,X_test1.CLASS)
           cm=cm/len(X_test1.CLASS)
           fig = plt.figure(figsize=(10, 8))
           sns.heatmap(cm*100, annot=True,annot_kws={"size": 16})
           st.pyplot(fig) 

def manual_rename(df,col_names):
    if col_names==[]:
        return df
    name=""
    name = st.text_input('Manual Review of Unmatched Column Names wanted? Y/N',key=rn.randint(1,1000000))
    if name=="":
      st.warning('Please input an option.')
      st.stop()
    if name in "nN":
        st.warning('No Manual Review Wanted')
        return df
    if col_names==[]:
        return df
    options=set(my_dict.values())    
    options.add("Other- Enter Manually")
    for cols in list(df.columns):
        if cols in options:
            options.remove(cols)
    for col in col_names:
        name=""
        name = st.selectbox("Please select Appropraite Column Name for "+col,options)
        if name=="":
          st.warning('Please select option.')
          st.stop()
        if name=="Other- Enter Manually":
            name=""
            name = st.text_input('Manual Input Name for Column',key=rn.randint(1,1000000))
            if name=="":
                st.warning('Please provide column name.')
                st.stop()
        if name in options:
            options.remove(name)
        df.rename(columns={col: name},inplace = True)                         
    return df                             

#@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def faker_gn(sample_size):
    data_sample=data_creation(entries=sample_size)
    new_column_names,canonical_lst=column_matching(list(data_sample.columns))
    data_sample.columns=new_column_names
    data_sample=manual_rename(data_sample,  canonical_lst)   
    st.write(data_sample.head(5))
    dfA1=data_sample
    dfB1=data_sample
    dfA1["initials"] = (dfA1["First Name"].str[0]  + dfA1["Last Name"].str[0])
    dfB1["initials"] = (dfB1["First Name"].str[0]  + dfB1["Last Name"].str[0])
    dfA1["Date of Birth"] = dfA1["Date of Birth"].astype(str).str.replace('-', "")
    dfB1["Date of Birth"] = dfB1["Date of Birth"].astype(str).str.replace('-', "")
    dfA1["soc_sec_id"] = dfA1["Social Security Number"].str.replace('-', "")
    dfB1["soc_sec_id"] = dfB1["Social Security Number"].str.replace('-', "")
    dfA1['Address']=dfA1['Street Number']+" "+dfA1['Address1']+" "+dfA1['Address2']
    dfB1['Address']=dfB1['Street Number']+" "+dfB1['Address1']+" "+dfB1['Address2'] 
    features1=data1(dfA1,dfB1,"initials")
    return dfA1,dfB1,features1  
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def bloom_grams(df,grams=3,prime_numbers=[89,97]):
    def func(x):
      nonlocal grams,prime_numbers
      s=["0"]*max(prime_numbers)
      #padding
      copy_x=" "*(grams-1)+str(x)+" "*(grams-1)
      for index in range(grams-1,len(copy_x)):
        curr_str=copy_x[index-grams+1:index+1]
        val=hash(curr_str)
        for i in prime_numbers:
          s[val%i]="1"
      return "".join(s)
    return df.applymap(func)
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def features_encrypt(dfA,dfB,candidate_links):
    def cosine_sim(str1,str2):
      sim=0
      str1_l2=0
      str2_l2=0
      for i in range(len(str1)):
        if str1[i]=='1':
          str1_l2+=1
        if str2[i]=='1':
          str2_l2+=1
        if str1[i]=='1' and str2[i]=='1':
          sim+=1  
      div=math.sqrt(str1_l2*str2_l2) 
      return sim/div    
    merge_list=[]
    columns_model=["First Name","Last Name","Suburb","State","Address","Date of Birth"]
    for index in range(len(candidate_links)):
      i=candidate_links[index]
      merge_list.append([])
      for col in columns_model:
          val1=dfA.loc[i[0],col]
          val2=dfB.loc[i[1],col]
          merge_list[-1].append(cosine_sim(val1,val2))
    merge_pd=pd.DataFrame(merge_list,columns=columns_model)
    return merge_pd






with faker_data:
    name = st.text_input('Do you want to test Feature Selection model on Faker Data? Y/N')
    if name=="":
      st.warning('Please input an option.')
      st.stop()
    if name in "nN":
        st.warning('No Testing on Faker Data Required')
    else:    
        st.markdown('<p class="font2">Running Model on Faker Data</p>', unsafe_allow_html=True)
        sample_size=st.slider("What would be the sample size of Fake Data?", min_value=100,max_value=5000,value=500,step=100)
        dfA1,dfB1,featuressour =faker_gn(sample_size)
        features1=featuressour.copy()
        if models=="Gradient Boosting" or models=="Logistic Regression":
            input1=features1
        else:
            L_fake = applier.apply(df=features1)
            input1=L_fake 
        #Match_Rate=st.slider("What would be the Probabilty Match Rate?", min_value=0.1,max_value=1.0,value=0.1,step=0.1)
        Display_Matches=st.slider("How many top Macthes to display?", min_value=1,max_value=10,value=5,step=1)    
        features1['Match']=model_final.predict_proba(input1)[:,1]
        features1.reset_index(inplace=True)
        features1=features1[features1["level_0"]!=features1["level_1"]]
        show=features1[features1['Match']>=0]

        st.markdown('<p class="font2">Number of Matches for Unencrypted Data', unsafe_allow_html=True)
        st.write(len(show)//2)
        show.sort_values(['Match'],ascending=False,inplace=True)
        show=show.reset_index(drop=True)
        display=0
        for i in range(0,len(show),2):
              display+=1
              if display==Display_Matches:
                  break
              st.markdown('<p class="font3">Probability of Matching', unsafe_allow_html=True)
              f1=show.iloc[i]
              st.write(f1["Match"]*100)
              d1=dfA1.iloc[[show['level_0'].values[i],show['level_1'].values[i]]]
              st.write(d1)

        st.markdown('<p class="font2">Encryption Part</p>', unsafe_allow_html=True)
        grams=st.slider("What would be the N Grams for Bloom Filter Encryption?", min_value=2,max_value=5,value=3,step=1)
        prime_numbers = st.multiselect("Please select Prime Numbers for Hashing", [83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149])
        if not prime_numbers:
          st.warning('Please input prime number.')
          st.stop()
        st.success('Thank you for inputting prime numbers.')
        st.write(grams)
        st.write(prime_numbers)
        dfA1_hash=bloom_grams(dfA1,grams,prime_numbers)
        st.markdown('<p class="font2">Encrypted Fake Data</p>', unsafe_allow_html=True)
        st.write(dfA1_hash.head(5)) 
        dfB1_hash=dfA1_hash.copy()
        
        cand_links=candidate_links_func(dfA1,dfB1,"initials")
        encrypt_features_df=features_encrypt(dfA1_hash,dfB1_hash,cand_links)
        encrypt_features_df.index=cand_links
        encrypt_features_df1=encrypt_features_df.copy()
        if models=="Gradient Boosting" or models=="Logistic Regression":
            encrypt_input1=encrypt_features_df1
        else:
            L_fake = applier.apply(df=encrypt_features_df1)
            encrypt_input1=L_fake 

        encrypt_features_df1['Match']=model_final.predict_proba(encrypt_input1)[:,1]
        encrypt_features_df1.reset_index(inplace=True)
        encrypt_features_df1=encrypt_features_df1[encrypt_features_df1["level_0"]!=encrypt_features_df1["level_1"]]
        show=encrypt_features_df1[encrypt_features_df1['Match']>=0]
        st.markdown('<p class="font2">Top Matches for Encrypted Data', unsafe_allow_html=True)
        show.sort_values(['Match'],ascending=False,inplace=True)
        show=show.reset_index(drop=True)
        display=0
        for i in range(0,len(show),2):
              display+=1
              if display==Display_Matches:
                  break
              st.markdown('<p class="font3">Probability of Matching', unsafe_allow_html=True)
              f1=show.iloc[i]
              st.write(f1["Match"]*100)
              d1=dfA1.iloc[[show['level_0'].values[i],show['level_1'].values[i]]]
              st.write(d1)
 

with upload_data:
    st.markdown('<p class="font2">Apply Record Linkage on your own Dataset</p>', unsafe_allow_html=True)
    uploaded_file= st.file_uploader("Choose your First File. File should be in csv format")
    while not uploaded_file:
        st.warning('Please upload File.')
        st.stop()      
    while  "csv" not in uploaded_file.type:
        st.warning('Please upload a csv File.')
        st.stop()
    st.success("Thank you for inputting First File.")    
    dataframe1 = pd.read_csv(uploaded_file)
    uploaded_file1= st.file_uploader("Choose your Second File. File should be in csv format")
    if not uploaded_file1:
        st.warning('Please upload File.')
        st.stop()
    while "csv" not in uploaded_file1.type:
        st.warning('Please upload a csv File.')
        st.stop()
    st.success("Thank you for inputting Second File.")      
    dataframe2 = pd.read_csv(uploaded_file1)    
    
    st.markdown('<p class="font3">Column Name Matching on Dataset 1', unsafe_allow_html=True)
    new_column_names,canonical_lst=column_matching(list(dataframe1.columns))
    dataframe1.columns=new_column_names
    dataframe1=manual_rename(dataframe1,  canonical_lst)   
    st.write(dataframe1.head(5))

    st.markdown('<p class="font3">Column Name Matching on Dataset 2', unsafe_allow_html=True)
    new_column_names,canonical_lst=column_matching(list(dataframe2.columns))
    dataframe2.columns=new_column_names
    dataframe2=manual_rename(dataframe2,  canonical_lst)   
    st.write(dataframe2.head(5))
    
    dataframe1["initials"] = (dataframe1["First Name"].str[0]  + dataframe1["Last Name"].str[0])
    dataframe2["initials"] = (dataframe2["First Name"].str[0]  + dataframe2["Last Name"].str[0])
    
    
    dataframe1_hash=dataframe1.copy()
    dataframe2_hash=dataframe2.copy()
    cand_links=candidate_links_func(dataframe1,dataframe2,"initials")
    name = st.text_input('Is the uploaded data encrypted? Y/N',key=rn.randint(1,1000000))
    if name=="":
      st.warning('Please input an option.')
      st.stop()
if name in "nN":
    with encryption_required:
        st.markdown('<p class="font2">Encrypting Your Dataset</p>', unsafe_allow_html=True)
        grams=st.slider("What would be the N Grams for Bloom Filter Encryption?", min_value=2,max_value=5,value=3,step=1, key = "2arb")
        prime_numbers = st.multiselect("Please select Prime Numbers for Hashing", [83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149], key = "44arbit")
        if not prime_numbers:
          st.warning('Please input prime number.')
          st.stop()
        st.success('Thank you for inputting prime numbers.')
        st.write(grams)
        st.write(prime_numbers)
        dataframe1_hash=bloom_grams(dataframe1,grams,prime_numbers)
        dataframe2_hash=bloom_grams(dataframe2,grams,prime_numbers)
        st.markdown('<p class="font2">Encrypted DataFrame 1</p>', unsafe_allow_html=True)
        st.write(dataframe1_hash.head(5))     
        st.markdown('<p class="font2">Encrypted DataFrame 2</p>', unsafe_allow_html=True)
        st.write(dataframe2_hash.head(5))    

with encrypion_not_required:
    encrypt_features_df= features_encrypt(dataframe1_hash,dataframe2_hash,cand_links)
    encrypt_features_df.index=cand_links
    encrypt_features_df1=encrypt_features_df.copy()
    if models=="Gradient Boosting" or models=="Logistic Regression":
        encrypt_input1=encrypt_features_df1
    else:
        L_fake = applier.apply(df=encrypt_features_df1)
        encrypt_input1=L_fake 

    encrypt_features_df1['Match']=model_final.predict_proba(encrypt_input1)[:,1]
    encrypt_features_df1.reset_index(inplace=True)
    encrypt_features_df1=encrypt_features_df1[encrypt_features_df1["level_0"]!=encrypt_features_df1["level_1"]]
    show=encrypt_features_df1[encrypt_features_df1['Match']>=0]
    st.markdown('<p class="font2">Top Matches for Encrypted Data', unsafe_allow_html=True)
    Display_Matches=st.slider("How many top Macthes to display?", min_value=1,max_value=10,value=5,step=1,key='klop') 
    show.sort_values(['Match'],ascending=False,inplace=True)
    show=show.reset_index(drop=True)
    display=0
    for i in range(0,len(show)):
          display+=1
          if display==Display_Matches:
              break
          st.markdown('<p class="font3">Probability of Matching', unsafe_allow_html=True)
          f1=show.iloc[i]
          st.write(f1["Match"]*100)
          d1=dataframe1.iloc[[show['level_0'].values[i]]]
          d2=dataframe2.iloc[[show['level_1'].values[i]]]
          d3=pd.concat([d1, d2], join="inner")
          st.write(d3)        


    
    st.markdown("[Scroll up](#capstone-project)",unsafe_allow_html=True)      
