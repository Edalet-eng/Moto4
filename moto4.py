from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd
import warnings
import pickle
import time
import PIL

warnings.filterwarnings(action = 'ignore')

df = pd.read_csv('data.csv')

interface = st.container()

with interface:
    st.title(body = 'Avtomobil Özəlliklərini daxil edin')
   
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

        
        
        
        
        
        
        
        
        rənglənib, vuruğu_var = st.columns(spec = [1, 1])
    
    with rənglənib:
        rənglənib = st.selectbox(label = 'rənglənib?', options =df.rənglənib.str.capitalize().unique().tolist())
        
        
    with vuruğu_var:
        vuruğu_var = st.selectbox(label = 'Vuruğu var? ', options = df.vuruğu_var.str.capitalize().unique().tolist())
    
    
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)
        
   
    st.subheader(body = 'Avtomobil təchizatı')
    
    
    lehimli_disk, abs_, lyuk, yağış_sensoru,dəri_salon = st.columns(spec = [1, 1, 1, 1, 1])
   
    with lehimli_disk:
        lehimli_disk = st.selectbox(label = 'Yüngül lehimli disklər' , options=df.lehimli_disk.str.capitalize().unique().tolist())
        
    with abs_:
        abs_ = st.selectbox(label = 'abs' , options=df.abs_.str.capitalize().unique().tolist())
    
    with lyuk:
        lyuk = st.selectbox(label = 'lyuk' , options=df.lyuk.str.capitalize().unique().tolist())
        
    with yağış_sensoru:
        yağış_sensoru = st.selectbox(label = 'yağış_sensoru' , options=df.yağış_sensoru.str.capitalize().unique().tolist())
        
    with dəri_salon:
        dəri_salon = st.selectbox(label = 'dəri_salon' , options=df.dəri_salon.str.capitalize().unique().tolist())
        
    
    st.markdown(body = '***')
    
    
    
    
    
    
    st.subheader(body = 'elaveler')
    mərkəzi_qapanma,park_radarı, kondisioner, oturacaqların_isidilməsi,  = st.columns(spec = [1, 1, 1, 1])
    
    with mərkəzi_qapanma:
        mərkəzi_qapanma = st.selectbox(label = 'mərkəzi_qapanma' , options=df.mərkəzi_qapanma.str.capitalize().unique().tolist())
    
    with park_radarı:
        park_radarı = st.selectbox(label = 'park_radarı' , options=df.park_radarı.str.capitalize().unique().tolist())
        
    with kondisioner:
        kondisioner = st.selectbox(label = 'kondisioner' , options=df.kondisioner.str.capitalize().unique().tolist())
    
    with oturacaqların_isidilməsi:
        oturacaqların_isidilməsi = st.selectbox(label = 'oturacaqların_isidilməsi' , options=df.oturacaqların_isidilməsi.str.capitalize().unique().tolist())
        
        
    
    ksenon_lampalar, arxa_görüntü_kamerası, yan_pərdələr, oturacaqların_ventilyasiyası = st.columns(spec = [1, 1, 1, 1])
    
    with ksenon_lampalar:
        ksenon_lampalar = st.selectbox(label = 'ksenon_lampalar' , options=df.ksenon_lampalar.str.capitalize().unique().tolist())
    
    with arxa_görüntü_kamerası:
        arxa_görüntü_kamerası = st.selectbox(label = 'arxa_görüntü_kamerası' , options=df.arxa_görüntü_kamerası.str.capitalize().unique().tolist())
        
    with yan_pərdələr:
        yan_pərdələr = st.selectbox(label = 'yan_pərdələr' , options=df.yan_pərdələr.str.capitalize().unique().tolist())
    
    with oturacaqların_ventilyasiyası:
        oturacaqların_ventilyasiyası = st.selectbox(label = 'oturacaqların_ventilyasiyası' , options=df.oturacaqların_ventilyasiyası.str.capitalize().unique().tolist())
        
        
        
        
        
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
        'abs': [abs_],
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
    
    with open('pipe_model.pickle', 'rb') as pickled_model:
        
        model = pickle.load(pickled_model)
    
    if st.button('Predict'):
        cars_price = model.predict(input_features)

        with st.spinner('Sending input features to model...'):
            time.sleep(2)

        st.success('Prediction is ready')
        time.sleep(1)
        st.markdown(f'### Car\'s estimated price is:  {cars_price} AZN')