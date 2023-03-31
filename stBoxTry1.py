#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import pyplot as p
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

    

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=550,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)
#%%
# print('38',canvas_result.image_data)
# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
    a=canvas_result.image_data
    print('43 a.shape=',a.shape)
    b=a[:,:,0]
    print('46 b.shape=',b.shape)
    fig1=p.figure(1)
    p.imshow(b)
    st.pyplot(fig1)   #p.imshow(b);p.show()
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    c=canvas_result.json_data["objects"]
    print('53 c=',c)
    df=pd.json_normalize(c)
    print()
    print('56 df=',df)
    print()
    print('58 df.top=',df.top)
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)
    # print('48',canvas_result.image_data)
#%%
st.write('b=',b)

