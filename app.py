import streamlit as st
import numpy as np
import torch
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFilter
from matplotlib import colormaps

st.set_page_config(page_title="Bone Fracture Detection", page_icon="🦴", layout="centered")

with st.sidebar:
    st.header("About")
    st.write("This application uses a Vision Transformer (ViT) model fine-tuned on X-ray images to detect bone fractures.")
    st.write("When an image is uploaded, the model not only predicts the presence of a fracture but also generates an attention map indicating which regions of the image most strongly influenced its decision.")
    st.write("---")
    st.write("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")

st.title("🦴 Bone Fracture Detection Web App")
st.write("Upload an X-Ray image to detect if there is a bone fracture. This app uses a pre-trained Vision Transformer model from Hugging Face.")

@st.cache_resource
def load_model():
    # Loading a pre-trained model for bone fracture detection from HuggingFace
    # Model: prithivMLmods/Bone-Fracture-Detection or Hemgg/bone-fracture-detection-using-xray
    pipe = pipeline("image-classification", model="prithivMLmods/Bone-Fracture-Detection")
    return pipe


def make_attention_box(pipe, image, class_index):
    """Return an approximate attention box for the predicted class."""
    model = pipe.model
    model.eval()
    inputs = pipe.image_processor(images=image, return_tensors="pt")
    device = next(model.parameters()).device
    pixel_values = inputs["pixel_values"].to(device).requires_grad_(True)

    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        logits = model(pixel_values=pixel_values).logits
        logits[0, class_index].backward()

    if pixel_values.grad is None:
        return image.copy(), None, None

    saliency = pixel_values.grad.detach().abs().mean(dim=1)[0]
    saliency -= saliency.min()
    if saliency.max() == 0:
        return image.copy(), None, None

    saliency = saliency / saliency.max()
    saliency_image = Image.fromarray((saliency.cpu().numpy() * 255).astype(np.uint8))
    saliency_image = saliency_image.resize(image.size, Image.Resampling.BILINEAR)
    saliency_image = saliency_image.filter(ImageFilter.GaussianBlur(radius=5))
    saliency_array = np.asarray(saliency_image)
    threshold = max(1, int(np.percentile(saliency_array, 85)))
    ys, xs = np.where(saliency_array >= threshold)

    if len(xs) == 0:
        max_y, max_x = np.unravel_index(np.argmax(saliency_array), saliency_array.shape)
        box_width = max(1, image.width // 4)
        box_height = max(1, image.height // 4)
        left = max(0, int(max_x) - box_width // 2)
        top = max(0, int(max_y) - box_height // 2)
        right = min(image.width - 1, left + box_width)
        bottom = min(image.height - 1, top + box_height)
    else:
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())

        mask_area = (right - left + 1) * (bottom - top + 1)
        if mask_area > image.width * image.height * 0.75:
            max_y, max_x = np.unravel_index(np.argmax(saliency_array), saliency_array.shape)
            box_width = max(1, image.width * 2 // 5)
            box_height = max(1, image.height * 2 // 5)
            left = max(0, int(max_x) - box_width // 2)
            top = max(0, int(max_y) - box_height // 2)
            right = min(image.width - 1, left + box_width)
            bottom = min(image.height - 1, top + box_height)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle((left, top, right, bottom), outline="red", width=5)
    
    # Create Heatmap overlay
    cmap = colormaps.get_cmap("jet")
    heatmap_array = cmap(saliency.cpu().numpy())
    heatmap_img = Image.fromarray((heatmap_array[:, :, :3] * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(image.size, Image.Resampling.BILINEAR)
    blended_heatmap = Image.blend(image.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=0.4).convert("RGB")

    return annotated, (left, top, right, bottom), blended_heatmap

st.write("Loading the best model (this might take a moment on the first run)...")
pipe = load_model()
st.success("Model loaded successfully!")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded X-ray", width="stretch")
    
    st.write("Analyzing...")
    
    # Run the model
    with st.spinner("Classifying image..."):
        results = pipe(image, top_k=2)
        
    st.subheader("Results:")
    for result in results:
        label = result['label']
        score = result['score'] * 100
        
        # Color coding the output based on typical fracture labels
        if 'fracture' in label.lower() and 'not' not in label.lower():
            st.error(f"**{label.capitalize()}** ({score:.2f}% confidence)")
        else:
            st.success(f"**{label.capitalize()}** ({score:.2f}% confidence)")

    predicted_index = pipe.model.config.label2id.get(results[0]["label"])
    if predicted_index is not None:
        with st.spinner("Estimating the area the model focused on..."):
            annotated_image, box, heatmap = make_attention_box(pipe, image, int(predicted_index))

        if box is not None:
            st.subheader("Approximate Area of Interest")
            col1, col2 = st.columns(2)
            with col1:
                st.image(annotated_image, caption="Red box: model attention estimate", width="stretch")
            with col2:
                if heatmap is not None:
                    st.image(heatmap, caption="Heatmap: model focus intensity", width="stretch")
            st.warning("These visualizations are AI attention estimates. They are not a medical diagnosis or a validated fracture location detector.")
        else:
            st.info("The model could not produce a reliable attention estimate for this image.")
