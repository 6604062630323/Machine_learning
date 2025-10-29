import streamlit as st
from PIL import Image
import torch
import numpy as np
from annoy import AnnoyIndex
from torchvision import models, transforms
st.image("Machine_Banner.png", use_container_width=True)

st.markdown(
    """
    <style>
    /* เปลี่ยนพื้นหลังทั้งหมด */
    .stApp {
        background-color: #ffffff;
    }

    header[data-testid="stHeader"] {
        background-color: #74aefd; 
    }

    /* ตั้งค่าสีตัวอักษรทั้งหมดเป็นดำ */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: black !important;
    }

    .banner {
        width: 100%;
        height: 250px;
        object-fit: cover;
        border-radius: 0px 0px 20px 20px;
    }
    /* พื้นหลังของปุ่ม upload */
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        background-color: white;
        color: white;
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 20px;
    }

    /* สีข้อความในปุ่ม */
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] div div span {
        color: black !important;
    }

    /* เปลี่ยนสีขอบเมื่อ hover */
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #1E90FF;
        background-color: #f5f9ff;
    }

    
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Image Similarity Search Demo")

# โหลด Annoy index และข้อมูล
EMBEDDING_DIM = 2048
annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular')
annoy_index.load("annoy_model.ann")
image_paths = np.load("image_paths.npy", allow_pickle=True)
labels = np.load("labels.npy", allow_pickle=True)

# โหลด ResNet feature extractor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_model = models.resnet50(pretrained=True)
feature_model = torch.nn.Sequential(*list(feature_model.children())[:-1])
feature_model.to(device)
feature_model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_embedding(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = feature_model(img_t).squeeze().cpu().numpy()
    return vec / np.linalg.norm(vec)

def search_similar(image, n_results=5):
    query_vec = get_embedding(image)
    indices = annoy_index.get_nns_by_vector(query_vec, n_results)
    return [(image_paths[i], labels[i]) for i in indices]

LOCAL_DATA_PATH = "data_split"
def fix_path(path):
    return path.replace("/content/drive/MyDrive/data_split", LOCAL_DATA_PATH)


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Searching for similar images")
    results = search_similar(img)
    for path, label in results:
        fixed_path = fix_path(path)
        st.image(fixed_path, caption=f"{label}", use_container_width=True)

