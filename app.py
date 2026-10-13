import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, UnidentifiedImageError


# -----------------------------------
# 1. Application configuration
# -----------------------------------

st.set_page_config(
    page_title="AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

MODEL_PATH = os.path.join(
    "models",
    "waste_classifier.keras"
)

CLASS_NAMES_PATH = os.path.join(
    "models",
    "class_names.json"
)

IMAGE_SIZE = (128, 128)


# -----------------------------------
# 2. Application heading
# -----------------------------------

st.title("♻️ AI-Based Waste Classification")

st.write(
    "Upload an image of waste to get a prediction "
    "using a Convolutional Neural Network (CNN)."
)

st.info(
    "This model recognizes six categories: "
    "cardboard, glass, metal, paper, plastic, and trash."
)


# -----------------------------------
# 3. Check whether training is complete
# -----------------------------------

if (
    not os.path.exists(MODEL_PATH)
    or not os.path.exists(CLASS_NAMES_PATH)
):
    st.warning(
        "The trained model has not been found. "
        "Please run train_model.py first."
    )

    st.code("python train_model.py")
    st.stop()


# -----------------------------------
# 4. Load the model and class names
# -----------------------------------

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


try:
    model = load_trained_model()
    class_names = load_class_names()

except Exception as error:
    st.error(
        "Unable to load the trained model. "
        "Please check the training output."
    )
    st.exception(error)
    st.stop()


# -----------------------------------
# 5. Upload an image
# -----------------------------------

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------------
# 6. Predict the category
# -----------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Image selected for classification",
            use_container_width=True
        )

        if st.button("Classify Waste"):

            # Resize the image to match model input.
            resized_image = image.resize(IMAGE_SIZE)

            # Convert the image into a NumPy array.
            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            # Add batch dimension: (1, 128, 128, 3).
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Get prediction probabilities.
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(predictions)
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = float(
                predictions[predicted_index]
            )

            # Show the result.
            st.success(
                f"Predicted category: "
                f"{predicted_class.title()}"
            )

            st.metric(
                "Model confidence",
                f"{confidence * 100:.2f}%"
            )

            st.caption(
                "Confidence is the model's estimated score, "
                "not a guarantee that the prediction is correct."
            )

            # Show scores for all categories.
            st.subheader("Category Scores")

            for index, category in enumerate(class_names):
                score = float(predictions[index])

                st.write(
                    f"{category.title()}: "
                    f"{score * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, score))
                )

    except (UnidentifiedImageError, OSError):
        st.error(
            "The uploaded file could not be opened "
            "as an image. Please upload a valid image."
        )

    except Exception as error:
        st.error("An error occurred during prediction.")
        st.exception(error)