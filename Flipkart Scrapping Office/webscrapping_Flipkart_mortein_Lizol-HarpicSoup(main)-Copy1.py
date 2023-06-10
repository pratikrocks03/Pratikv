#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns%matplotlib inline
import re
import time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
#from gazpacho import Soup
import datefinder
import datetime
from dateutil.relativedelta import relativedelta
#from datetime import datetime


# In[2]:


header = {
  'Accept': '*/*',
  'Accept-Language': 'en-US,en;q=0.9',
  'Connection': 'keep-alive',
  'Content-Type': 'application/json',
  'Cookie': 'T=TI165224556239300130121196300902181168905413091029128241491504644103; _pxvid=0f4990ec-d0e8-11ec-afba-4b6e6f685066; pxcts=3f7008d5-e55e-11ec-be0b-74525a767948; s_cc=true; AMCVS_17EB401053DAF4840A490D4C%40AdobeOrg=1; AMCV_17EB401053DAF4840A490D4C%40AdobeOrg=-227196251%7CMCIDTS%7C19150%7CMCMID%7C56565202329804192053069826446062803447%7CMCAAMLH-1655100103%7C12%7CMCAAMB-1655100103%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1654502503s%7CNONE%7CMCAID%7CNONE; Network-Type=4g; _px3=510170f740ec7a58cedc8cdb23e0f0f190200f6147e901114448d68451b83063:jzH8hZ8GAPmyXydXHKr4+VfZ3wPytyFz4Z8X3B2DvFgu0sHs0Uu2MATISSU8BNsn+mD0Ieke2hP6Igic/KIu1w==:1000:hVRTPxI1E0V96sUpGiny+efK3/Ot9uVYpn5W6Ht4blMy/PTi4n9jYPlHPb9aORigMmgCBrt8QO/9AlBjPY3RrwFDrbvUQdYZUURrEqswZRrRsDKeTvQBdo80evmz8v68GO9ExaTOTsEY55mCTFYZv86NFgJKUIC/agWeztKyA0qXyuzdqWHeF2l4Us8LNp2AGiCBqXDV8up6aFe6RXATRA==; SN=VIE8A827454110484CA20BB28E101E6858.TOKB092030172C840A4946BDC3B990F05EB.1654498390.LO; S=d1t14GA4nbD8/Cz8KPyIqBkk/P2/28YaxQ/sAeR0jsdY0eO28G1iYWzTxlN1muz1Jz5UzfVJQgU+7XXsq3S+1JDOKNw==; s_sq=%5B%5BB%5D%5D; S=d1t14XEE4ADMhangiPz8ePz8/cmcs6voZdEppp8jSsQJHwS79JmK8N80mP4L6Rw35fJWBSjUAK4U0/o8DsFirOZh4CA==; SN=VIE8A827454110484CA20BB28E101E6858.TOKB092030172C840A4946BDC3B990F05EB.1654498390.LO',
  'Origin': 'https://www.flipkart.com',
  'Referer': 'https://www.flipkart.com/',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-site',
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36',
  'X-User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 FKUA/website/42/website/Desktop',
  'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Windows"'
}


# In[12]:


# Mortein
asin=[]

for i in range(1,4):
    url="https://www.flipkart.com/search?q=mortein&as=on&as-show=on&otracker=AS_Query_HistoryAutoSuggest_1_2_na_na_na&otracker1=AS_Query_HistoryAutoSuggest_1_2_na_na_na&as-pos=1&as-type=HISTORY&suggestionId=good+night&requestId=06c9a0be-3d06-4e7d-b3b9-7a1aea7a5e14&as-searchtext=go&page={}".format(i)
    #print(url)
    page=requests.get(url,headers=header)
    if page.status_code==200:
        soup=BeautifulSoup(page.content)
        for d in soup.findAll('div', attrs={'class':"_1AtVbE col-12-12"}):
            for a in d.findAll('div',attrs={'class':"_13oc-S"}):
                print(a)
                #asin.append(d['data-id'])
                #print(asin)
    else:
        print("Error")


# In[3]:


# for Lizol
asin=[]

for i in range(1,2):
    url="https://www.flipkart.com/search?q=sanifresh&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off=go&page={}".format(i)
    #print(url)
    page=requests.get(url,headers=header)
    if page.status_code==200:
        soup=BeautifulSoup(page.content)
        for d in soup.findAll('div', attrs={'class':"_1AtVbE col-12-12"}):
            for a in d.findAll('div',attrs={'class':"_13oc-S"}):
                print(a)
                #asin.append(d['data-id'])
                #print(asin)
    else:
        print("Error")


# In[13]:


asin


# In[3]:


#mortein
Asin3=['IRPEU9J7MREU5AHS','IRPG7S3R2ZK5TRA5','MVRG7ZUPHU4VTJMD','IRPF8YDCDSRYVX3G','IRPF8YDCYAUJXCQF','IRPEUHGGGYYBGWMN',
      'EIKFVVCY9HJYMQYF','MVRGFBMGFXJDB3QW','MVRG7ZUTUNYYDD3E','IRPF8YDCBNXXZZHS','IRPG5KDHSG9TJ5R9','MOCFJQDXZW9E4QPC']


# In[22]:


Asin2=['BCRGY4GSNBGF7DNV','BCRET7WHT9WDDFYG','BCRETEEZDFWJTVWB','BCRET7WHG8SKCKFD','BCRG384BFEYHHJYC','BCRFTHKVX9UYEBGG',
      'BCRETEEDVGYQGA3N','BCRFHGKJCHARZHDA','BCRETEEZH6E32ZMY','BCRET7WGAKGXDZ9D','BCRGY4GSYZ5BF3RJ','BCRFTHJZRHADBFFY',
      'KSCF6Y3AKRKHAT2Y','BCRGC2PMAJZCHM9H','BCRG5FP3DZC2RVGD','BCRGBB5RFYGTEZKM','BCRFV4G2JVJWFBJG','BCRGB355FJSWRZMC',
      'BCRG5FP3PPZBDFHD','BCRFBRNHUKFDV2BV','BCRFMTHNMZ2VCXKU','BCRGB4H9GXQTPHWG','BCRFUGZSAD5GQAV4','BCRG8SWGNDPFGNNY']


# In[3]:


# Good night
Asin2=['MOVFNVDHDSYQNVPG','MOVFNVDHP7F9JSFT','MOVFNVDHD87WZCRG','MVRFWYBZWEZTXFXF','MVRG4WHGVHSVYSEB','MOVFMF9FKGD7S9QY',
     'MOCFX8WS5MHTUDRF','IRPFW9ZGWG5SEFMX','MOCG2DQHT4WGFAYA','MOCFJQC5E3RQEZX9','IRPEUHGMUQPHAVYF','MOCFJQC55JGGGVZN',
      'IRPFVN44YGVTWHHM','BLAG7XEYHFGAU5PX','IRPEU9J7FJMJPWRH','MVRGESGGFMMKFXXG','MVRFFFSCEQBTBBC5','MVRFWYBZRGV4YGE4',
        'MOVGEKHNYFAVXRHS','MOVFN74YUJUKKCRX','MOCFVTX86JMADUK7','MOCFJQC5XSN34V34','MOCGHFBUFGR28Z5F','MOCFZT5YPGVECREN',
      'MOCG2DQV3ZW6NJ4M','MNTG9A9HYYHS7P4J','MVRFWYBZFFK3AC27','MOVFNVDHQVPBNNNK','MVRG9Y8W8FQZH9RM','MVRGF6JK2YA64VKH',
      'MVRGESHSEWW22CDP','MVRFSXSHDYK8ZZBD','MNTG9A9H5JQ7Z8ZG','BCHFMMYCKSB8PRNZ','IRPEU9J7VHEKUHHS','MVRG3ZNVXHBGGFAJ',
      'MVRFZJQDZGRJP8SW','MOVG42QYWD2ZPC58','MOVGESHB6YXHGRXU']


# In[5]:


# Vanish
asin=["SRMGB2YFZSXEQMHW","SRMF67MGD9YWBQZW","SRMGH44RWAGHRZ2H","SRMG7RXGPBKTYRPH",
     "SRMGCZQD6M74T9ZS","SRMGB8J5SWQJZAJ2","SRMGAYCNE2NA4UGT","SRMGKFMFNVC2R8TK",
     "WSPFMHJRFC2PQKRE","LDGFJNZZD85VZRW4","WSPFMPEHGCEVZYKR","WSPFGDGBYCTXCFBH",
     "WSPFZ6QPWYPZ3CYJ","FWRFM9ZVKVHBZSJE","SRMGG47NBWGJ3GDZ","SRMFC7D6ZUFVZNSJ",
     "SRMGAYEMGTSCEZJN","SRMGH4UHQTE93K5M","SRMG9WN6ZZHMWUWX","SRMGFA4HNEP5Z6QZ",
     "LDGFMGJYGBMHCMQ6","WSPFGDG6M4TVQACR","WSPFMHJPD6FTHTRT","LDGEZVAUZUSZCYZ5",
     "LDGGCEJTHR7EGFVG","EIKFVVCY9HJYMQYF","SRMGBYKZN9FNPF9W","SRMGCYV9UASXBRV9"]


# In[5]:


#Lizol
asin=["BCRET7WHT9WDDFYG","BCRG384BFEYHHJYC","BCRFKGWHQ5J7GADU","BCRET7WGAKGXDZ9D"]


# In[13]:


# Godrej Hit
Asin = ["IRPEU9J7KEYB2NGQ","IRPEU9J7FQPHZQVB","IRPFN9FYZFFJRMRE","IRPEU9J7BMTADYVH",
       "IRPGYZQ9AZKKGU6D","IRPGAFNPSVYK2Q7W","EIKFMF8EVZ4K8VFT","ETYFXSFRGC6GVMKY",
       "IRPG3U5MGCPRE83T","RATGH82UGSEEZHYZ"]


# In[23]:


# Domex
asin = ["TCNGADHGGNSZCV8B", "TCNGADHGHJUWCDNH", "TCNETM7YQCCSNSFH", "TCNF34ZR7VCHUZ9B",
       "TCNFMGBBYAP9Q5WW", "TCNFMZ7YSZSMTUGT", "TCNF34ZRSBUJKZGN", "TCNG53JNDEZNKCFF",
       "TCNFZ5CG2EU3JSJ5", "TCNFZ4H3YXRUNUFQ", "TCNG57NTM9NYN3EZ", "TCNG9UJFHKTCYVPU",
       "TCNGYQZ9PH3WTBYQ", "TCNFEYH5X5AS82HY", "TCNFZ3EQH3EZAYHH", "TCNFUGZAHNG8NVAV"]


# In[3]:


# Sanifresh
asin1= ["TCNG4SBXSQGYTYP2", "TCNFZJEGJA57VHHZ", "TCNG4SCFS4TDYQSX","TCNFG4G2NP58CKVT","TCNFMGJFXZNDRQZW",
       "TCNFPEYZYNKEXH9G", "TCNEU6ZFTXJ6RJAT", "TCNFWYGX5KYWKHEU", "TCNEU6ZFXWJMTJNQ","TCNG57QZZGEFRAWW",
       "TCNG2Y5UGRUHY6KG", "TCNG24HFB67MSPGD","TCNFZTUT6NYF3JTR"]


# In[3]:


# Pee Safe

Asin1 = ["TCNF94B5WZQYHYNK", "TCNGBNFRNY4XZVPG", "TCNG3VSU6SMGF7V3", "TCNG3UXFZHMY2UJ8", "TCNF94B7CYT44GQ4",
        "TCNG3VTPYGSEGZMW", "TCNG3UX6UUPD4GV7", "TCNF94CKVSKG2SGZ", "TCNGF8BUVAGAGYXB", "TCNGAH3MFN8JKYT2",
        "TCNG3VTB4RXRPHHC", "TCNF53YGCKKZZRYU", "TCNF94B23JN4HJKK", "TCNF94EZDUAZJRMX", "TCNGNP7HG38W8HBG",
        "TCNF2TVHG8MFGNMA"]


# In[13]:


# Pee Buddy

Asin2 = ["TCNGGHJRQRKWJZ2C", "TCNGGGHQTGYKC7FX", "TCNFWTSPBMVYQMH9", "TCNGGGHGGGAEWF5P","TCNFWTSPSGCYHUBZ",
        "TCNFWTJPY92PCFFT", "TCNGGGMKSVADSBFH"]


# In[24]:


output=[]


# In[14]:


page1=[]
for j in Asin:
    try:
        for i in range(1,6):
            url="https://www.flipkart.com/lizol-disinfectant-surface-floor-cleaner-floral/product-reviews/itmc7b54d3fb6adf?pid={}&lid=LSTBCRG384BFEYHHJYCDR7TTW&aid=overall&certifiedBuyer=false&sortOrder=MOST_RECENT&page={}".format(j,i)
            print(url)
            page=requests.get(url,headers=header)
            if page.status_code==200:
                soup=BeautifulSoup(page.content)
                page1.append(soup)
            else:
                print("Error")
    except:
        continue


# In[15]:


'''page1=[]

for i in range(1,2):
    url="https://www.flipkart.com/search?q=good+night&as=on&as-show=on&otracker=AS_Query_HistoryAutoSuggest_1_2_na_na_na&otracker1=AS_Query_HistoryAutoSuggest_1_2_na_na_na&as-pos=1&as-type=HISTORY&suggestionId=good+night&requestId=06c9a0be-3d06-4e7d-b3b9-7a1aea7a5e14&as-searchtext=go&page={}".format(i)
    print(url)
    page=requests.get(url,headers=header)
    if page.status_code==200:
        soup=BeautifulSoup(page.content)
        page1.append(soup)
    else:
        print("Error")
'''


# In[16]:


soup=page1


# In[8]:


allpn=[]
allid=[]
alln=[]
alld=[]
allt=[]
alls=[]
allr=[]
allc=[]

for i in soup:
       for d in i.findAll('div', attrs={'class':"col _2wzgFH K0kLPL"}):
            id2=d.find('p',attrs={'class':"_2mcZGG"})
            ID=id2['id']
            productname=i.find('a',attrs={'class':"s1Q9rs _2qfgz2"})
            Date = d.select_one('._2sc7ZR:contains("ago")')
            Title = d.find('p', attrs={'class':"_2-N8zT"})
            Rating = d.find("div",class_=["_3LWZlK _1BLPMq","_3LWZlK _32lA32 _1BLPMq"])
            Review2 = d.find('div', attrs={'class':"t-ZTKy"})
            Review3=Review2.find('div', attrs={'class':None})
            Review=Review3.find('div')
            
            #print(Title)
            #print(productname)
            #print(ID)
            
            if ID is not None:
                allid.append(ID)
            else:
                allid.append("unknown-ID")
            if productname is not None:
                allpn.append(productname.text)
            else:
                allpn.append("unknown-product")
            if Date is not None:
                alld.append(Date.text)
            else:
                alld.append("unknown-product")
            if Title is not None:
                allt.append(Title.text)
            else:
                allt.append("unknown-product")
            if Review is not None:
                allc.append(Review.text)
            else:
                allc.append("unknown-product")
            if Rating is not None:
                allr.append(Rating.text)
            else:
                allr.append("unknown-product")
   


# In[9]:


a={'ID':allid,'Product Name':allpn,'Date':alld,'Title':allt,'Rating':allr,'Review':allc}
output=pd.DataFrame.from_dict(a, orient='index')
output=output.transpose()


# In[10]:


output.info()


# In[11]:


output


# In[18]:


from deep_translator import GoogleTranslator
from deep_translator import exceptions as excp


# In[19]:


def translate(text):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
    except (excp.NotValidPayload, excp.NotValidLength) as e:
        translated = text
    return translated


# In[20]:


output['Date'] = output['Date'].apply(translate)


# In[21]:


output


# In[12]:


output.to_excel("Good Knight May flipkart may-23 scrape.xlsx")


# In[92]:


#def date(a):
    matches = datefinder.find_dates(a)
    for match in matches:
          return match


# In[93]:


#output['Date']=output['Date'].apply(date)


# In[94]:


#output['Title'] = output['Title'].replace(r'\n','', regex=True) 


# In[96]:


#output['Review'] = output['Review'].replace(r'\n','', regex=True) 


# In[97]:


# def rate(a):
    matches = re.sub('de 5 estrelas', '', a)
    return matches


# In[98]:


output['Rating']=output['Rating'].apply(rate)


# In[107]:


output


# In[102]:


d = datetime.datetime(2022,7,2)


# In[ ]:





# In[87]:


#len(set(output["ID"]))


# In[30]:


#output.to_excel("Lizol IN July 2022 trial.xlsx")


# In[47]:


#output2=pd.read_excel("Lizol IN July 2022 trial.xlsx")


# In[64]:


##output2=output


# In[48]:


#output2.head()


# In[49]:


#d = datetime.datetime(2022,8,2)


# In[50]:


#def ago_do_date(ago):
    value, unit = re.search(r'(\d+) (\w+) ago', ago).groups()
    if not unit.endswith('s'):
        unit += 's'
    delta = relativedelta(**{unit: int(value)})
    x=d - delta
    print(d - delta)
    return x
    


# In[51]:


#output2['Date2']=output2['Date'].apply(ago_do_date)


# In[52]:


#output2.info()


# In[53]:


#output2['Date2'] = pd.to_datetime(output2.Date2, format='%Y-%m-%d')


# In[54]:


#output2.head()


# In[55]:


#output2=output2[output2["Date2"].dt.year == 2022]
output2=output2[output2["Date2"].dt.month == 7]


# In[56]:


#output2.tail()


# In[27]:


#output2.info()


# In[28]:


output2.to_excel("Mortein Flipkart June 2022 Scrape.xlsx")
#df_main_april.to_excel("FairyAprilAmazonScrape.xlsx")

