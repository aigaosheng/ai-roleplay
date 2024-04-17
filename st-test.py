import streamlit as st
import numpy as np
import pandas as pd
import time

df1 = {
    "A1": [1,2],
    "A2":[10, 11],
}
df2 = {
    "A1": [10,20],
    "A2":[20, 30],
}
# Add rows to a dataframe after
# showing it.
element = st.dataframe(df1)
element.add_rows(df2)

# Add rows to a chart after
# showing it.
element = st.line_chart(df1)
element.add_rows(df2)

# Show a spinner during a process
with st.spinner(text='In progress'):
  time.sleep(3)
  st.success('Done')

# Show and update progress bar
bar = st.progress(50)
time.sleep(3)
bar.progress(100)

st.balloons()
# st.snow()
st.toast('Mr Stay-Puft')
# st.error('Error message')
# st.warning('Warning message')
# st.info('Info message')
# st.success('Success message')
# st.exception(e)