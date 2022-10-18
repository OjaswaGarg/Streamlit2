import streamlit as st
import pandas as pd
import sys
import subprocess
import math
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',package])
import_or_install("faker")
import_or_install("PIL")
import_or_install("tqdm")
from tqdm import tqdm
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

header=st.container()
dataset=st.container()
featurest=st.container()
model_training=st.container()
faker_data=st.container()
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
      compare.string('given_name', 'given_name', method='cosine', label="given_name")
      compare.string('surname', 'surname', method='cosine', label="surname")
      compare.string('suburb', 'suburb', method='cosine', label="suburb")
      compare.string('state', 'state', method='cosine', label="state")
      compare.string('address', 'address', method='cosine', label="address")
      compare.string("date_of_birth","date_of_birth",method='cosine', label="date_of_birth")
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

st.sidebar.title("Interactive") 


models=st.sidebar.selectbox("How would you like the data to be modeled?",("Gradient Boosting", "Logistic Regression", "Weak Supervision"))

metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def data():
    dfA, dfB, true_links = load_febrl4(return_links=True)
    dfA["initials"] = (dfA["given_name"].str[0]  + dfA["surname"].str[0])
    dfB["initials"] = (dfB["given_name"].str[0]  + dfB["surname"].str[0])
    dfA["date_of_birth"] = dfA["date_of_birth"].str.replace('-', "")
    dfB["date_of_birth"] = dfB["date_of_birth"].str.replace('-', "")
    dfA["soc_sec_id"] = dfA["soc_sec_id"].str.replace('-', "")
    dfB["soc_sec_id"] = dfB["soc_sec_id"].str.replace('-', "")
    dfA['address']=dfA['street_number']+" "+dfA['address_1']+" "+dfA['address_2']
    dfB['address']=dfB['street_number']+" "+dfB['address_1']+" "+dfB['address_2'] 
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
    st.markdown('<p class="font2">Data from Record Linkage Package</p>', unsafe_allow_html=True)
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
    st.markdown('<p class="font3">Input train Dataset', unsafe_allow_html=True)
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
        st.markdown('<p class="font2">Applying Gradient Boosting to Model</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="font2">Applying Logistic Regression to Model</p>', unsafe_allow_html=True)
    
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
       st.markdown('<p class="font2">Applying Snorkel to Model</p>', unsafe_allow_html=True)
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


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def faker_gn(sample_size):
    data_sample=data_creation(entries=sample_size)
    dfA1=data_sample
    dfB1=data_sample
    dfA1["initials"] = (dfA1["given_name"].str[0]  + dfA1["surname"].str[0])
    dfB1["initials"] = (dfB1["given_name"].str[0]  + dfB1["surname"].str[0])
    dfA1["date_of_birth"] = dfA1["date_of_birth"].astype(str).str.replace('-', "")
    dfB1["date_of_birth"] = dfB1["date_of_birth"].astype(str).str.replace('-', "")
    dfA1["soc_sec_id"] = dfA1["soc_sec_id"].astype(str).str.replace('-', "")
    dfB1["soc_sec_id"] = dfB1["soc_sec_id"].astype(str).str.replace('-', "")
    dfA1['address']=dfA1['street_number']+" "+dfA1['address_1']+" "+dfA1['address_2']
    dfB1['address']=dfB1['street_number']+" "+dfB1['address_1']+" "+dfB1['address_2']
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
    for index in tqdm(range(len(candidate_links))):
      i=candidate_links[index]
      val1=dfA.loc[i[0]].tolist()
      val2=dfB.loc[i[1]].tolist()
      merge_list.append([])
      merge_list[-1].append(cosine_sim(val1[0],val2[0]))
      merge_list[-1].append(cosine_sim(val1[1],val2[1]))
      merge_list[-1].append(cosine_sim(val1[5],val2[5]))
      merge_list[-1].append(cosine_sim(val1[7],val2[7]))
      merge_list[-1].append(cosine_sim(val1[11],val2[11]))
      merge_list[-1].append(cosine_sim(val1[8],val2[8]))
    return merge_list
with faker_data:
    st.markdown('<p class="font2">Running Model on Faker Data</p>', unsafe_allow_html=True)
    sample_size=st.slider("What would be the sample size of Fake Data?", min_value=100,max_value=5000,value=500,step=100)
    dfA1,dfB1,featuressour =faker_gn(sample_size)
    features1=featuressour.copy()
    if models=="Gradient Boosting" or models=="Logistic Regression":
        input1=features1
    else:
        L_fake = applier.apply(df=features1)
        input1=L_fake 
    Match_Rate=st.slider("What would be the Probabilty Match Rate?", min_value=0.1,max_value=1.0,value=0.1,step=0.1)
    Display_Matches=st.slider("How many top Macthes to display?", min_value=1,max_value=10,value=5,step=1)    
    features1['Match']=model_final.predict_proba(input1)[:,1]
    features1.reset_index(inplace=True)
    features1=features1[features1["level_0"]!=features1["level_1"]]
    show=features1[features1['Match']>=Match_Rate]
    
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
    st.write(grams)
    st.write(prime_numbers)
    dfA1_hash=bloom_grams(dfA1,grams,prime_numbers)
    st.markdown('<p class="font2">Encrypted Fake Data</p>', unsafe_allow_html=True)
    st.write(dfA1_hash.head(5)) 
    st.write(dfA1_hash.columns) 
    dfB1_hash=dfA1_hash.copy()
    cand_links=candidate_links_func(dfA1,dfB1,"initials")
    merge_list= features_encrypt(dfA1_hash,dfB1_hash,cand_links)
    encrypt_features_df=pd.DataFrame(merge_list,columns=list(featuressour.columns))
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
    show=encrypt_features_df1[encrypt_features_df1['Match']>=Match_Rate]
    
    st.markdown('<p class="font2">Number of Matches for Encrypted Data', unsafe_allow_html=True)
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
    
    
    
    st.markdown("[Scroll up](#capstone-project)",unsafe_allow_html=True)      
