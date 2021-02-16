import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
import itertools
import re 
import datetime 

st.title('Home assistant helper')
ha_directory="/usr/share/hassio/homeassistant/www/deepstack"
archive_directory="/mnt/hdd/hassio/deepstack"

def paginator(label, items, items_per_page=10, on_sidebar=True):
    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_format_func = lambda i: "Page %s" % i
    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    return itertools.islice(enumerate(items), min_index, max_index)

header_list = ["filename", "x_min", "y_min", "x_max", "y_max", "label", "confidence"]
df = pd.read_csv(ha_directory + '/labels.csv', header=None, names=header_list) 

df['datetime'] = df.filename.str.replace(r'.*_20([0-9-]*)',r'20\1').str.replace(r'([0-9-]*)_([0-9-]*).*',r'\1 \2')
df['crop_file_prefix'] = df.filename.str.replace(r'.*(deepstack_object.*)_nobox.jpg',r'\1')

labels=df.groupby(['label']).size().sort_values(ascending=False)

labelsticked=st.sidebar.selectbox('Show?', labels.keys())

images = []
prev = ""
for index, row in df.iterrows():
    if row['crop_file_prefix'] == prev:
        counter += 1
    else:
        counter = 1
    prev = row['crop_file_prefix']
    if labelsticked in row['label']: 
        objtype = "object"
        if "face" in labelsticked:
            objtype="face"
        filename = row['crop_file_prefix'] + "_" + objtype + "_" + labelsticked + "_{}".format(counter) + ".jpg"
        ff = archive_directory + "/" + filename
        if (os.path.isfile(ff)):
            images.append(ff)
            continue
        ff = ha_directory + "/" + filename
        if (os.path.isfile(ff)):
            images.append(ff)
            continue
        
if images:
    images.sort(key=os.path.getctime, reverse=True)        
    image_iterator = paginator("Select a page", images, 36, False)
    indices_on_page, images_on_page = map(list, zip(*image_iterator))
    st.image(images_on_page, width=100, height=100, caption=indices_on_page)

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected:', option

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("Woohoo!")

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")