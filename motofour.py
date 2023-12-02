import streamlit as st
import pandas as pd
import warnings
import pickle
import time
import streamlit as st
import plotly.graph_objects as go
import PIL
from PIL import Image


df = pd.read_csv('data.csv')


interface = st.container()

with interface:


    st.title('Dataset haqqında')

    st.subheader('Datasetə daxil etdiyiniz məlumatlar aşağıdakılardır:')

    st.markdown("<p style='font-size: 20px;'>1. Marka - Avtomobilin markası və ya istehsalçısı</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>2. Model - Avtomobilin xüsusi modeli</p>",unsafe_allow_html = True)
    

    st.markdown("<p style='font-size: 20px;'>4. Yanacaq novu - Avtomobilin istifadə etdiyi yanacaq növü</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>5. Ötürücü - Avtomobilin ötürücü növü</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>6. Ban növü - Avtomobilin ban növü</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>7. Sürət qutusu - Avtomobildəki sürət qutusu (məsələn: avtomatik, mexaniki)</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>8. Yürüş - Avtomobilin getdiyi ümumi məsafə</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>9. Buraxılış ili - Avtomobilin istehsal olunduğu il</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>10. Rəng - Avtomobilin rəngi</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>11. Hansı bazar üçün - Avtomobil hansı bazar üçün yığılıb</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>12. Mühərrikin həcmi - Avtomobilin mühərrikinin ölçüsü(sm³)</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>13. Mühərrikin gücü - Avtomobilin mühərrikinin at gücündə gücü (HP)</p>",unsafe_allow_html = True)
    
    
    st.markdown("<p style='font-size: 20px;'>16. Rənglənib? - Avtomobil rənglənib və ya rənglənməyib</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>17. Vuruğu var? - Avtomobilin vuruğu var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>18. Yüngül lehimli disklər - Yüngül lehimli disklər var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>19. ABS - ABS var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>20. Lyuk - Lyuk var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>21. Yağış sensoru - Yağış sensoru var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>22. Dəri salon - Dəri salon var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>23. Mərkəzi qapanma - Mərkəzi qapanma var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>24. Park radarır - Park radarı var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>25. Kondisioner - Kondisioner var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>26. Oturacaqların isidilməsi - Oturacaqların isidilməsi var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>27. Ksenon lampalar - Ksenon lampalar var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>28. Arxa görüntü kamerası - Arxa görüntü kamerası var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>29. Yan pərdələr - Yan pərdələr var və ya yoxdur</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>29. Oturacaqların ventilyasiyası - Oturacaqların ventilyasiyası var və ya yoxdur</p>",unsafe_allow_html = True)


    st.info(f'Məlumat toplusu {df.shape[0]} sıra və {df.shape[1]} sütundan ibarətdir. ')

  