from PIL import Image
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
import sklearn
import sqlite3
df=pd.read_csv('data.csv')
 
def label_encoder_process(data_frame=None):
    for i in data_frame.columns:
        lb = LabelEncoder()
        data_frame[i]=lb.fit_transform(data_frame[i])
    return data_frame
    
interface = st.container()


with interface:
    
    label_encoder = LabelEncoder()

    
    marka_encoding = label_encoder.fit_transform(df['marka'].str.capitalize())
    marka_mapping = {name: value for name, value in zip(df.marka.str.capitalize().tolist(), marka_encoding)}

    model_encoding = label_encoder.fit_transform(df['model'])
    model_mapping = {name: value for name, value in zip(df.model.str.capitalize().tolist(), model_encoding)}


    yanacaq_novu_encoding = label_encoder.fit_transform(df['yanacaq_novu'])
    yanacaq_novu_mapping = {name: value for name, value in zip(df.yanacaq_novu.str.capitalize().tolist(), yanacaq_novu_encoding)}

    ötürücü_encoding = label_encoder.fit_transform(df['ötürücü'])
    ötürücü_mapping = {name: value for name, value in zip(df.ötürücü.str.capitalize().tolist(), ötürücü_encoding)}

    ban_növü_encoding = label_encoder.fit_transform(df['ban_növü'])
    ban_növü_mapping = {name: value for name, value in zip(df.ban_növü.str.capitalize().tolist(), ban_növü_encoding)}

    sürətlər_qutusu_encoding = label_encoder.fit_transform(df['sürətlər_qutusu'])
    sürətlər_qutusu_mapping = {name: value for name, value in zip(df.sürətlər_qutusu.str.capitalize().tolist(), sürətlər_qutusu_encoding)}

    rəng_encoding = label_encoder.fit_transform(df['rəng'])
    rəng_mapping = {name: value for name, value in zip(df.rəng.str.capitalize().tolist(), rəng_encoding)}

    hansı_bazar_encoding = label_encoder.fit_transform(df['hansı_bazar_üçün_yığılıb'])
    hansı_bazar_mapping = {name: value for name, value in zip(df.hansı_bazar_üçün_yığılıb.str.capitalize().tolist(), hansı_bazar_encoding)}

   
   
    st.title(body = 'Avtomobil Özəlliklərini daxil edin')
    
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)
    
   
    marka , model = st.columns(spec = [1, 1])
    

    with marka:
        marka = st.selectbox(label = 'Marka', options =df['marka'].str.capitalize().sort_values().unique().tolist())
    
    with model:
        model = st.selectbox(label = 'Model', options =df[df['marka'].str.capitalize() == marka]['model'].str.capitalize().sort_values().unique().tolist())  
        
    st.markdown(body = '***')
    
    
    yanacaq_novu, ötürücü, ban_növü, sürətlər_qutusu = st.columns(spec = [1, 1, 1, 1])
        
    with yanacaq_novu:
        yanacaq_novu = st.selectbox(label = 'Yanacaq növü', options =df.yanacaq_novu.str.capitalize().unique().tolist())
    
    with ötürücü:
        ötürücü = st.selectbox(label = 'Ötürücü', options =df.ötürücü.str.strip().str.capitalize().unique().tolist())

    with ban_növü:
        ban_növü = st.selectbox(label = 'Ban növü', options =df.ban_növü.str.capitalize().unique().tolist())
        
    with sürətlər_qutusu:
        sürətlər_qutusu = st.selectbox(label = 'Sürətlər qutusu', options = df.sürətlər_qutusu.str.strip().str.capitalize().unique().tolist())
        
        
    
    yürüş = st.number_input(label = 'Yürüş (km)', value = 0, step = 1000 )
    button_text = 'Dəyəri göndərin'
 
    st.markdown(body = '***')
    
    buraxılış_ili = st.slider(label='İl',min_value = int(df.buraxılış_ili.min()),
                              max_value= int(df.buraxılış_ili.max()),value = int(df.buraxılış_ili.mean()))
        
    rəng, hansı_bazar_üçün_yığılıb = st.columns(spec = [1, 1])
    
    with rəng:
         rəng = st.selectbox(label = 'Rəng', options =df.rəng.str.strip().str.capitalize().sort_values().unique().tolist())
            
    with hansı_bazar_üçün_yığılıb:
         hansı_bazar_üçün_yığılıb = st.selectbox(label = 'Hansı bazar üçün yığılıb', options =df.hansı_bazar_üçün_yığılıb.str.capitalize().sort_values().unique().tolist())
            
    st.markdown(body = '***')
    
    
    mühərrik_hecmi, mühərrik_gucu = st.columns(spec = [1, 1])
    
    with mühərrik_hecmi:
        mühərrik_hecmi = st.number_input(label = 'Mühərrikin həcmi (sm³)', value = 0, step = 50 )
        button_text = 'Send values'
    
    with mühərrik_gucu:
        mühərrik_gucu = st.number_input(label = 'Mühərrikin gücü, a.g.', value = 0, step = 1 )
        button_text = 'Send values' 

        
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)

     
    st.subheader(body = 'Vəziyyət')
    
    rənglənib, vuruğu_var = st.columns(spec = [1, 1])
    
    with rənglənib:
        rənglənib = st.radio(label = 'Rənglənib? ', options = ['rənglənib', 'rənglənməyib'], horizontal = True)
        
    with vuruğu_var:
        vuruğu_var = st.radio(label = 'Vuruğu var? ', options = ['vuruğu var', 'vuruğu yoxdur'], horizontal = True)
    
    
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)
        
   
    st.subheader(body = 'Avtomobil təchizatı')
    
    
    lehimli_disk, abs, lyuk, yağış_sensoru,dəri_salon = st.columns(spec = [1, 1, 1, 1, 1])
   
    with lehimli_disk:
        lehimli_disk = st.checkbox(label = 'Yüngül lehimli disklər')
        
    with abs:
        abs = st.checkbox(label = 'ABS')
    
    with lyuk:
        lyuk = st.checkbox(label = 'Lyuk')
        
    with yağış_sensoru:
        yağış_sensoru = st.checkbox(label = 'Yağış sensoru')
        
    with dəri_salon:
        dəri_salon = st.checkbox(label = 'Dəri salon')
        
    
    st.markdown(body = '***')
 
        
    
    mərkəzi_qapanma,park_radarı, kondisioner, oturacaqların_isidilməsi,  = st.columns(spec = [1, 1, 1, 1])
    
    with mərkəzi_qapanma:
        mərkəzi_qapanma = st.checkbox(label = 'Mərkəzi qapanma')
    
    with park_radarı:
        park_radarı = st.checkbox(label = 'Park radarı')
        
    with kondisioner:
        kondisioner = st.checkbox(label = 'Kondisioner')
    
    with oturacaqların_isidilməsi:
        oturacaqların_isidilməsi = st.checkbox(label = 'Oturacaqların isidilməsi')
        
        
    st.markdown(body = '***')
    
    
    ksenon_lampalar, arxa_görüntü_kamerası, yan_pərdələr, oturacaqların_ventilyasiyası = st.columns(spec = [1, 1, 1, 1])
    
    with ksenon_lampalar:
        ksenon_lampalar = st.checkbox(label = 'Ksenon lampalar')
    
    with arxa_görüntü_kamerası:
        arxa_görüntü_kamerası = st.checkbox(label = 'Arxa görüntü kamerası')
        
    with yan_pərdələr:
        yan_pərdələr = st.checkbox(label = 'Yan pərdələr')
    
    with oturacaqların_ventilyasiyası:
        oturacaqların_ventilyasiyası = st.checkbox(label = 'Oturacaqların ventilyasiyası')
        
        
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)


    rənglənib_encoding = {'rənglənməyib':1,'rənglənib':0}
    vuruğu_var_encoding = {'vuruğu yoxdur':1,'vuruğu var':0}
       
    
    df['marka'] = marka_encoding
    df['model'] = model_encoding
    df['yanacaq_novu'] = yanacaq_novu_encoding
    df['ötürücü'] = ötürücü_encoding
    df['ban_növü'] = ban_növü_encoding
    df['sürətlər_qutusu'] = sürətlər_qutusu_encoding
    df['rəng'] = rəng_encoding
    df['hansı_bazar_üçün_yığılıb'] = hansı_bazar_encoding
    df['rənglənib'] = df['rənglənib'].replace(rənglənib_encoding)
    df['vuruğu_var'] = df['vuruğu_var'].replace(vuruğu_var_encoding)
    


    marka = marka_mapping[marka]
    model = model_mapping[model]
    yanacaq_novu = yanacaq_novu_mapping[yanacaq_novu]
    ötürücü = ötürücü_mapping[ötürücü]
    ban_növü = ban_növü_mapping[ban_növü]
    sürətlər_qutusu = sürətlər_qutusu_mapping[sürətlər_qutusu]
    rəng = rəng_mapping[rəng]
    hansı_bazar_üçün_yığılıb = hansı_bazar_mapping[hansı_bazar_üçün_yığılıb]
    rənglənib = rənglənib_encoding[rənglənib]
    vuruğu_var = vuruğu_var_encoding[vuruğu_var]
    
    lehimli_disk = int(lehimli_disk)
    abs = int(abs)
    lyuk = int(lyuk)
    yağış_sensoru = int(yağış_sensoru)
    mərkəzi_qapanma = int(mərkəzi_qapanma)
    park_radarı = int(park_radarı)
    kondisioner = int(kondisioner)
    oturacaqların_isidilməsi = int(oturacaqların_isidilməsi)
    dəri_salon = int(dəri_salon)
    ksenon_lampalar = int(ksenon_lampalar)
    arxa_görüntü_kamerası = int(arxa_görüntü_kamerası)
    yan_pərdələr = int(yan_pərdələr)
    oturacaqların_ventilyasiyası = int(oturacaqların_ventilyasiyası)

       

    
    input_features = pd.DataFrame({
        'marka': [marka],
        'model': [model],
        'ban_növü': [ban_növü],
        'rəng': [rəng],
        'sürətlər_qutusu': [sürətlər_qutusu],
        'ötürücü': [ötürücü],
        'hansı_bazar_üçün_yığılıb': [hansı_bazar_üçün_yığılıb],
        'yanacaq_novu': [yanacaq_novu],
        'vuruğu_var': [vuruğu_var],
        'rənglənib': [rənglənib],
        'lehimli_disk': [lehimli_disk],
        'abs': [abs],
        'lyuk': [lyuk],
        'yağış_sensoru': [yağış_sensoru],
        'mərkəzi_qapanma': [mərkəzi_qapanma],
        'park_radarı': [park_radarı],
        'kondisioner': [kondisioner],
        'oturacaqların_isidilməsi': [oturacaqların_isidilməsi],
        'dəri_salon': [dəri_salon],
        'ksenon_lampalar': [ksenon_lampalar],
        'arxa_görüntü_kamerası': [arxa_görüntü_kamerası],
        'yan_pərdələr': [yan_pərdələr],
        'oturacaqların_ventilyasiyası': [oturacaqların_ventilyasiyası],
        'buraxılış_ili': [buraxılış_ili],
        'yürüş': [yürüş],
        'mühərrik_hecmi': [mühərrik_hecmi],
        'mühərrik_gucu': [mühərrik_gucu]
        
    })
    
    

    st.subheader(body = 'Model Prediction')
    
    with open('saved_model.pickle', 'rb') as pickled_model:
        
        model = pickle.load(pickled_model)
    
    if st.button('Predict'):
        cars_price = model.predict(input_features)

        with st.spinner('Sending input features to model...'):
            time.sleep(2)

        st.success('Prediction is ready')
        time.sleep(1)
        st.markdown(f'### Car\'s estimated price is:  {cars_price} AZN')
    
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)
    # Yorumlar bazasına qoşulun
    conn = sqlite3.connect('yorumlar.db')
    cursor = conn.cursor()
    
    # Streamlit tətbiqini yaradın
    st.title('Yorumlar Tətbiqi')
    
    # Yorum əlavə etmə formunu tərtib edin
    yorum = st.text_area("Yorumunuzu burada daxil edin:")
    submit = st.button("Göndər")
    
    # Yorum göndərildikdə
    if submit:
        # Əlavə olunacaq yorumları bazaya yazın
        cursor.execute("INSERT INTO yorumlar (yorum) VALUES (?)", (yorum,))
        conn.commit()
        st.success("Yorumunuz uğurla əlavə edildi.")
   
      
      
