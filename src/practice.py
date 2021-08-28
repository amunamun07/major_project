import streamlit as st

st.title("HR System")

from PIL import Image

st.subheader("This is a sub")

image = Image.open("src/aa.png")
st.image(image, use_column_width=True)

st.write("writing a text here !!")

st.markdown("Markdown")

st.success("Sucess")

st.info("Information")

st.warning("Warning")

st.error("Error")

st.help("Help")

import numpy as np
import pandas as pd

dataframe = np.random.rand(10,20)
st.dataframe(dataframe)

#display chart
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
st.line_chart(chart_data)
