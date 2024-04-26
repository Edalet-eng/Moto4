import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import openai
import streamlit as st
import openai
import pickle 
import time
from googletrans import Translator



# Read the CSV file
df = pd.read_csv('last_data.csv')
df = df.dropna()
st.set_page_config(initial_sidebar_state='expanded',
                  layout='wide')
st.container()



st.markdown("[MOTO4 a get](https://moto4.vercel.app)")

st.header("Zəhmət olmasa məlumatları tam daxil edin!")
st.markdown("Qeyd: Databazada mövcud məlumatlara əsasən təhlil aparıldığını nəzərə almağınız xahiş olunur!", unsafe_allow_html=True)

st.markdown("---")


marka ,model= st.columns(spec=[1,1],gap='small')
with marka:
    marka = st.selectbox(label = 'Avtomobilin markasını daxil edin:', options =df['marka'].str.capitalize().sort_values().unique().tolist())


with model:
    model = st.selectbox(label = 'Avtomobilin modelini daxil edin:', options =df[df['marka'].str.capitalize() == marka]['model'].str.capitalize().sort_values().unique().tolist()) 
    

buraxilis_group,yürüş_group= st.columns(spec=[1,1],gap='small')

with buraxilis_group:
    buraxilis_group = st.selectbox(label = 'Buraxılış ili aralığını daxil edin:', options =df[df['marka'].str.capitalize() == marka][df['model'].str.capitalize() == model]['buraxilis_group'].sort_values().unique().tolist())


with yürüş_group:
    yürüş_group= st.selectbox("Yürüş aralığını daxil edin:",options=df[df['marka'].str.capitalize() == marka][df['model'].str.capitalize() == model][df['buraxilis_group'] == buraxilis_group]['yürüş_group'].sort_values().unique().tolist())
    


try:
    st.markdown("<h4 style='text-align: center;'>Seçilmiş avtomobilin verilmiş tarix üzrə ortalama qiymətini analiz edə bilərsiniz.</h4>", unsafe_allow_html=True)


    mean_values = df[df['marka'].str.capitalize() == marka][df['model'].str.capitalize() == model][df['buraxilis_group'] == buraxilis_group][df['yürüş_group'] == yürüş_group].groupby(['yeniləndi'])['qiymet'].mean()
    mean_values= pd.DataFrame(mean_values)


    fig1= px.line(mean_values, title='Maşın Qiymətləri',color_discrete_sequence=['#620985'])

    # Adding x-axis labels and ticks
    fig1.update_xaxes(
        tickformat="%b %Y",
        dtick="M1",
        ticklabelmode="period")

    fig2= px.area(mean_values, title='Maşın Qiymətləri',color_discrete_sequence=['#620985'])

    # Adding x-axis labels and ticks
    fig2.update_xaxes(
        tickformat="%b %Y",
        dtick="M1",
        ticklabelmode="period")

    # Display the plots in Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)

    
except:
    st.write("<h6 style='text-align: center;'>Nəticə yoxdur. Zəhmət olmasa filterləməni düzgün daxil edin!</h6>", unsafe_allow_html=True)
    
    
        
st.markdown("***")

# Sample data - replace this with your actual data
qutu = df[df['marka'].str.capitalize() == marka][df['model'].str.capitalize() == model]['sürətlər_qutusu'].unique()
size = [1]*100

ban = df[df['marka'].str.capitalize() == marka][df['model'].str.capitalize() == model]['ban_növü'].unique()
size_ban = [1] * len(ban)

muherrik = df[df['marka'].str.capitalize() == marka][df['model'].str.capitalize() == model]['mühərrik_hecmi'].unique()
size_muherrik = [1] * len(muherrik)

yanacaq = df[df['marka'].str.capitalize() == marka][df['model'].str.capitalize() == model]['yanacaq_novu'].unique()
size_yanacaq = [1] * len(yanacaq)

# Create a container for the charts
center_container = st.container()

center_container.markdown("<h4 style='text-align: center;'>Seçilmiş avtomobil haqqında spesifik məlumatlar</h4>", unsafe_allow_html=True)
center_container.markdown("<h6 style='text-align: center;'>Qeyd: Qrafikləri görmək üçün Marka və Model seçməyiniz kifayətdir!</h6>", unsafe_allow_html=True)
center_container.markdown("---")

center_container.markdown(f"<h6 style='text-align: center;'>{marka + model} markalı avtomobil</h6>", unsafe_allow_html=True)
center_container.markdown(" ")
center_container.markdown(" ")


# Divide the container layout into two columns
col1, col2 = center_container.columns(2)

# Create a horizontal bar chart for 'sürətlər_qutusu'
with col1:
    fig1 = go.Figure(go.Bar(y=size, x=qutu, orientation='v',marker_color='#A400DF'))
    fig1.update_layout(
        title="Sürət qutusu növü",
        yaxis=dict(showticklabels=False),
        width=400,  # Set width of the figure
        height=300
    )
    st.plotly_chart(fig1)
    

# Create a horizontal bar chart for 'ban_növü'
with col2:
    fig2 = go.Figure(go.Bar(y=size_ban, x=ban, orientation='v',marker_color='#A400DF'))
    fig2.update_layout(
        title="Ban növü",
        yaxis=dict(showticklabels=False),
        width=400,  # Set width of the figure
        height=300
    )
    st.plotly_chart(fig2)
    
    
center_container.markdown("***")
# Divide the container layout into two columns
col3, col4 = center_container.columns(2)

# Create a horizontal bar chart for 'mühərrik_hecmi'
with col3:
    data_muherrik_hecm = pd.DataFrame({"Mühərrik həcmi": [df[df['marka'].str.capitalize() == marka][df['model'].str.capitalize() == model]['mühərrik_hecmi'].unique()],})
    st.data_editor(data_muherrik_hecm,column_config={"Mühərrik həcmi": st.column_config.ListColumn(width="large")}, hide_index=True,height=250)   
    
# Create a horizontal bar chart for 'yanacaq_novu'
with col4:
    fig4 = go.Figure(go.Bar(y=size_yanacaq, x=yanacaq, orientation='v',marker_color='#A400DF'))
    fig4.update_layout(
        title="Yanacaq növü",
        yaxis=dict(showticklabels=False),
        width=400,  # Set width of the figure
        height=300
    )
    st.plotly_chart(fig4)
center_container.markdown("***")


center_container.markdown("[MOTO4 a get](https://moto4.vercel.app)")











st.sidebar.title("Məsləhətçi bot")
openai.api_key = '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'
translator = Translator()
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar.form("chat_form"):
    for message in st.session_state.messages:
        with st.sidebar.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.text_input("Sual ver:", key="user_input")
    submit_button = st.form_submit_button("Daxil et")
    car_info_button = st.sidebar.button("Avtomobiliniz haqqında məlumat al")
if submit_button:
    prompt_eng = translator.translate(prompt, src='az', dest='en')
    st.session_state.messages.append({"role": "user", "content": prompt_eng.text})

    with st.sidebar.chat_message("user"):
        st.markdown(prompt)

    with st.sidebar.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[{"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages],
                stream=True,):
            full_response += response.choices[0].delta.get("content", "")
                #message_placeholder.markdown(full_response + "▌")
        translation = translator.translate(full_response, dest='az').text
        message_placeholder.markdown(translation)
    st.session_state.messages.append({"role": "assistant", "content": translation})


if car_info_button:
    # Get the selected values for marka and model
    marka_value = marka
    model_value = model
    year_value = buraxılış_ili
    engine_value = mühərrik_hecmi

    # Create a message to send to the chatbot
    car_info_message = f"{marka_value}/{model_value} markalı avtomobilin üstün və zəif tərəfləri haqqında ətraflı məlumat ver."
    car_info_message_eng = translator.translate(car_info_message, src='az', dest='en')

    # Send the message to the chatbot
    st.session_state.messages.append({"role": "user", "content": car_info_message_eng.text})
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
            delta_content = response.choices[0].delta.get("content")
            if delta_content is not None:
                full_response += delta_content
                message_placeholder.markdown(f"Xahiş olunur, bir neçə saniyə gözləyin... ▌")
                #message_placeholder.markdown(translated_response + "▌")
                    
        translation = translator.translate(full_response, dest='az').text
        message_placeholder.markdown(translation)
    st.session_state.messages.append({"role": "assistant", "content": translation})