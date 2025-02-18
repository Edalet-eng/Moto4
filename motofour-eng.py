import streamlit as st
from deta import Deta
from PIL import Image
import openai
import pandas as pd
import pickle
import numpy as np
import sqlite3
import time
from sklearn.preprocessing import LabelEncoder
import sklearn
import streamlit as st
#st.image('587-161.png', use_column_width=True)
custom_icon_url = "6060.jpg"  
df=pd.read_csv('lastdata.csv')
st.set_page_config(page_icon=custom_icon_url,
                   layout='wide' , 
                  initial_sidebar_state="expanded")


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
    
    

    
    st.markdown(
    """
    <style>
        .button {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1;
        }
        
        button:hover {
            background-color: white;
            color: #3498db;
        }
        button:active{
            background-color: white:
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
    
    

        
        

    st.title('Enter vehicle specifications')

    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)

    marka , model ,ban_növü= st.columns(spec = [1, 1,1])


    with marka:
        marka = st.selectbox(label = 'Brand', options =df['marka'].str.capitalize().sort_values().unique().tolist())

    with model:
        model = st.selectbox(label = 'Model', options =df[df['marka'].str.capitalize() == marka]['model'].str.capitalize().sort_values().unique().tolist())  
        
    with ban_növü:
        ban_növü = st.selectbox(label = 'Ban type', options =df.ban_növü.str.capitalize().unique().tolist())

    st.markdown(body = '***')


    yanacaq_novu, ötürücü, sürətlər_qutusu ,yürüş= st.columns(spec = [1, 1, 1,1])

    with yanacaq_novu:
        yanacaq_novu = st.selectbox(label = 'Fuel type', options =df.yanacaq_novu.str.capitalize().unique().tolist())
    
    with ötürücü:
        ötürücü = st.selectbox(label = 'Gear', options =df.ötürücü.str.strip().str.capitalize().unique().tolist())
        
    with sürətlər_qutusu:
        sürətlər_qutusu = st.selectbox(label = 'Transmission', options = df.sürətlər_qutusu.str.strip().str.capitalize().unique().tolist())

    with yürüş:
         yürüş = st.number_input(label = 'KM', value = 0, step = 1000 )

    st.markdown(body = '***')

    buraxılış_ili = st.slider(label='Year',min_value = int(df.buraxılış_ili.min()),
                              max_value= int(df.buraxılış_ili.max()),value = int(df.buraxılış_ili.mean()))

    rəng, hansı_bazar_üçün_yığılıb,mühərrik_hecmi ,mühərrik_gucu= st.columns(spec = [1, 1,1,1])

    with rəng:
         rəng = st.selectbox(label = 'Color', options =df.rəng.str.strip().str.capitalize().sort_values().unique().tolist())
            
    with hansı_bazar_üçün_yığılıb:
         hansı_bazar_üçün_yığılıb = st.selectbox(label = 'For which market it is assembled', options =df.hansı_bazar_üçün_yığılıb.str.capitalize().sort_values().unique().tolist())

    with mühərrik_hecmi:
        mühərrik_hecmi = st.number_input(label = 'Engine capacity (sm³)', value = 0, step = 50 )
        button_text = 'Send values'
    
    with mühərrik_gucu:
        mühərrik_gucu = st.number_input(label = 'Engine power, a.g.', value = 0, step = 1 )
        button_text = 'Send values' 


    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)


    st.subheader(body = 'The situation')

    rənglənib, vuruğu_var = st.columns(spec = [1, 1])

    with rənglənib:
        rənglənib = st.radio(label = 'İs it colored? ', options = ['rənglənib', 'rənglənməyib'], horizontal = True)
        
    with vuruğu_var:
        vuruğu_var = st.radio(label = 'Does it have a stroke? ', options = ['vuruğu var', 'vuruğu yoxdur'], horizontal = True)


    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)


    st.subheader(body = 'Car supplies')


    lehimli_disk, abs, lyuk, yağış_sensoru,dəri_salon,mərkəzi_qapanma,park_radarı = st.columns(spec = [1, 1, 1, 1, 1,1,1])

    with lehimli_disk:
        lehimli_disk = st.checkbox(label = 'Alloy wheels')
        
    with abs:
        abs = st.checkbox(label = 'ABS')
    
    with lyuk:
        lyuk = st.checkbox(label = 'Lyuk')
        
    with yağış_sensoru:
        yağış_sensoru = st.checkbox(label = 'Rain sensor')
        
    with dəri_salon:
        dəri_salon = st.checkbox(label = 'Leather salon')

    with mərkəzi_qapanma:
        mərkəzi_qapanma = st.checkbox(label = 'Central locking')
    
    with park_radarı:
        park_radarı = st.checkbox(label = 'Parking radar')


    st.markdown(body = '***')


    ksenon_lampalar, arxa_görüntü_kamerası, yan_pərdələr, oturacaqların_ventilyasiyası,kondisioner,oturacaqların_isidilməsi = st.columns(spec = [1, 1, 1, 1,1,1])

    with ksenon_lampalar:
        ksenon_lampalar = st.checkbox(label = 'Xenon lamps')
    
    with arxa_görüntü_kamerası:
        arxa_görüntü_kamerası = st.checkbox(label = 'Rear view camera')
        
    with yan_pərdələr:
        yan_pərdələr = st.checkbox(label = 'Side curtains')
    
    with oturacaqların_ventilyasiyası:
        oturacaqların_ventilyasiyası = st.checkbox(label = 'Seat ventilation')
        
    with kondisioner:
        kondisioner = st.checkbox(label = 'Air conditioning')
    
    with oturacaqların_isidilməsi:
        oturacaqların_isidilməsi = st.checkbox(label = 'Seat heating')


    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)


    elave_melumat = st.text_area(
    "Write additional notes about your car")

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



    marka2 = marka_mapping[marka]
    model2 = model_mapping[model]
    yanacaq_novu2 = yanacaq_novu_mapping[yanacaq_novu]
    ötürücü2 = ötürücü_mapping[ötürücü]
    ban_növü2 = ban_növü_mapping[ban_növü]
    sürətlər_qutusu2 = sürətlər_qutusu_mapping[sürətlər_qutusu]
    rəng2 = rəng_mapping[rəng]
    hansı_bazar_üçün_yığılıb2 = hansı_bazar_mapping[hansı_bazar_üçün_yığılıb]
    rənglənib2 = rənglənib_encoding[rənglənib]
    vuruğu_var2 = vuruğu_var_encoding[vuruğu_var]

    lehimli_disk2 = int(lehimli_disk)
    abs2 = int(abs)
    lyuk2 = int(lyuk)
    yağış_sensoru2 = int(yağış_sensoru)
    mərkəzi_qapanma2 = int(mərkəzi_qapanma)
    park_radarı2 = int(park_radarı)
    kondisioner2 = int(kondisioner)
    oturacaqların_isidilməsi2 = int(oturacaqların_isidilməsi)
    dəri_salon2 = int(dəri_salon)
    ksenon_lampalar2 = int(ksenon_lampalar)
    arxa_görüntü_kamerası2 = int(arxa_görüntü_kamerası)
    yan_pərdələr2 = int(yan_pərdələr)
    oturacaqların_ventilyasiyası2 = int(oturacaqların_ventilyasiyası)




    input_features = pd.DataFrame({
        'marka': [marka2],
        'model': [model2],
        'ban_növü': [ban_növü2],
        'rəng': [rəng2],
        'sürətlər_qutusu': [sürətlər_qutusu2],
        'ötürücü': [ötürücü2],
        'hansı_bazar_üçün_yığılıb': [hansı_bazar_üçün_yığılıb2],
        'yanacaq_novu': [yanacaq_novu2],
        'vuruğu_var': [vuruğu_var2],
        'rənglənib': [rənglənib2],
        'lehimli_disk': [lehimli_disk2],
        'abs': [abs2],
        'lyuk': [lyuk2],
        'yağış_sensoru': [yağış_sensoru2],
        'mərkəzi_qapanma': [mərkəzi_qapanma2],
        'park_radarı': [park_radarı2],
        'kondisioner': [kondisioner2],
        'oturacaqların_isidilməsi': [oturacaqların_isidilməsi2],
        'dəri_salon': [dəri_salon2],
        'ksenon_lampalar': [ksenon_lampalar2],
        'arxa_görüntü_kamerası': [arxa_görüntü_kamerası2],
        'yan_pərdələr': [yan_pərdələr2],
        'oturacaqların_ventilyasiyası': [oturacaqların_ventilyasiyası2],
        'buraxılış_ili': [buraxılış_ili],
        'yürüş': [yürüş],
        'mühərrik_hecmi': [mühərrik_hecmi],
        'mühərrik_gucu': [mühərrik_gucu]

    })

  

    st.subheader(body = 'Model prediction')


    try:
        with open('saved_model.pickle', 'rb') as pickled_model:
            pred_model = pickle.load(pickled_model)
    except Exception as e:
        st.error(f"Something wrong: {e}")
        # Add more details or actions if necessary



 
    if st.button('Predict'):
        try:
            if df[df['model'] == model_mapping[model]]['model'].count() < 7:
                st.warning("The car price you enter may not be predictable because there is not enough information in the database")
            else:
                st.success('Hesablanır')
                time.sleep(1)
                st.markdown(f'### Estimated price for the car: {np.round(int(pred_model.predict(input_features)),-2)} AZN')
        except Exception as e:
            st.error(f"Yanlış əməliyyat: {e}")
    # Add more details or actions if necessary


    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)
    qiymet = np.round(int(pred_model.predict(input_features)),-2)    




#     # SQLite veritabanı ilə əlaqə yaratmaq
#     engine = create_engine('sqlite:///comment.db', echo=True)

#     # Modeli təyin etmək
#     Base = declarative_base()
#     class Comment(Base):
#         __tablename__ = 'comment'
#         id = Column(Integer, primary_key=True)
#         comment = Column(String)


#     # Veritabanını yaratmaq üçün
#     metadata = MetaData()
#     cars_table = Table('comment', metadata,
#                        Column('id', Integer, primary_key=True),
#                        Column('comment', String))
#     metadata.create_all(engine)


#     st.subheader(body = 'Comment')

#     # Yorum əlavə etmə formunu tərtib edin
#     yorum = st.text_area("Submit your comment:")
#     submit = st.button("Submit")


#     def yorum_elave_et(yorum):
#         new_comment = Comment(comment=yorum)
#         session = Session(bind=engine)
#         session.add(new_comment)
#         session.commit ()
#         session.close()



        
#     # Streamlit tətbiqindən gələn məlumatlarla əlavə etmə funksiyasını çağırmaq
#     if submit:
#         yorum_elave_et(yorum)
#         st.success("Comment added")






        
        
        
    st.sidebar.title("Advisor bot")
    openai.api_key = "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"
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar.form("chat_form"):
        for message in st.session_state.messages:
            with st.sidebar.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.text_input("Say something:", key="user_input")
        submit_button = st.form_submit_button("Enter")
        car_info_button = st.sidebar.button("Get general information about the entered car")
    if submit_button:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.sidebar.chat_message("user"):
            st.markdown(prompt)

        with st.sidebar.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


    if car_info_button:
        # Get the selected values for marka and model
        marka_value = marka
        model_value = model
        year_value = buraxılış_ili
        engine_value = mühərrik_hecmi

        # Create a message to send to the chatbot
        car_info_message = f"Briefly describe the strengths and weaknesses of the {year_value} {marka_value} / {model_value} with {engine_value} engines."

        # Send the message to the chatbot
        st.session_state.messages.append({"role": "user", "content": car_info_message})
        with st.sidebar.chat_message("user"):
            st.markdown(car_info_message)

        with st.sidebar.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


 
