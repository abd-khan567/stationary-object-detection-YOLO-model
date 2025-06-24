import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

# # turn off Streamlit’s automatic file watcher to avoid introspection errors
# st.set_option("server.fileWatcherType", "none")
# st.set_option("server.runOnSave", False)

# Load the trained model (best.pt)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Prices for each item
prices = {
    'sharpener': 20,
    'pencil': 10,
    'scale': 30,
    'eraser': 15
}

st.title('🖍️ Stationary Item Detection & Bill Calculator')

uploaded_file = st.file_uploader("📸 Upload an image of stationery items", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Inference
    results = model(image)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    img_np = np.array(image)

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_np, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    st.image(img_np, caption='Detected Items', use_container_width=True)

    # Bill calculation
    detected_items = [model.names[int(cls)] for cls in labels]
    bill = 0
    st.subheader('🧾 Itemized Bill:')
    for item in set(detected_items):
        qty = detected_items.count(item)
        st.write(f"{item.capitalize()} × {qty} = {prices[item]*qty} PKR")
        bill += prices[item] * qty

    st.success(f"💰 Total: {bill} PKR")