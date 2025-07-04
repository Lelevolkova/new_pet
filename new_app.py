# import os
# import json
# import torch
# import streamlit as st
# import torchvision.transforms as transforms
# from PIL import Image
# from torchvision.models import resnet50, ResNet50_Weights
# from torch.nn.functional import cosine_similarity
# from geopy.distance import geodesic
# import folium
# from streamlit_folium import st_folium
# from base64 import b64encode

# # === Параметры ===
# DATASET_DIR = "/Users/lolovolkova/Desktop/pet_app_airi/id_pets"
# LOCATIONS_PATH = "/Users/lolovolkova/Desktop/pet_app_airi/animal_locations.json"
# TOP_N = 10

# # === Устройство ===
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # === Модель и трансформации ===
# @st.cache_resource
# def load_model():
#     model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(device)
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])
#     return model, transform

# model, transform = load_model()

# # === Извлечение эмбеддингов ===
# def get_embedding(image: Image.Image):
#     image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = model(image_tensor)
#     return embedding.squeeze(0)

# # === Поиск похожих животных ===
# def find_similar_animals(uploaded_images, lat, lon):
#     similarity_by_animal = {}
#     target_embeddings = [get_embedding(img) for img in uploaded_images]

#     for animal_id in os.listdir(DATASET_DIR):
#         animal_path = os.path.join(DATASET_DIR, animal_id)
#         if not os.path.isdir(animal_path):
#             continue

#         max_sim = -1
#         for fname in os.listdir(animal_path):
#             img_path = os.path.join(animal_path, fname)
#             try:
#                 img = Image.open(img_path)
#                 emb = get_embedding(img)
#                 for target_emb in target_embeddings:
#                     sim = cosine_similarity(target_emb.unsqueeze(0), emb.unsqueeze(0)).item()
#                     if sim > max_sim:
#                         max_sim = sim
#             except:
#                 continue

#         # if max_sim > 0:
#         #     similarity_by_animal[animal_id] = max_sim
#         if max_sim > 0.6:
#             similarity_by_animal[animal_id] = max_sim

#     with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
#         locations = json.load(f)

#     top_animals = sorted(similarity_by_animal.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
#     nearby, faraway, others = [], [], []

#     for animal_id, sim in top_animals:
#         if animal_id in locations:
#             coords = locations[animal_id]
#             dist = geodesic((lat, lon), (coords["lat"], coords["lon"])).km
#             first_img = get_first_image(animal_id)
#             info = (animal_id, coords, sim, dist, first_img)
#             if dist <= 1.0:
#                 nearby.append(info)
#             else:
#                 faraway.append(info)

#     others_ids = set(similarity_by_animal.keys()) - set([a[0] for a in top_animals])
#     for animal_id in others_ids:
#         if animal_id in locations:
#             coords = locations[animal_id]
#             dist = geodesic((lat, lon), (coords["lat"], coords["lon"])).km
#             first_img = get_first_image(animal_id)
#             others.append((animal_id, coords, similarity_by_animal[animal_id], dist, first_img))

#     return nearby, faraway, others

# def get_first_image(animal_id):
#     folder = os.path.join(DATASET_DIR, animal_id)
#     for file in os.listdir(folder):
#         if file.lower().endswith((".jpg", ".jpeg", ".png")):
#             return os.path.join(folder, file)
#     return None

# def image_to_base64(path):
#     with open(path, "rb") as img_file:
#         return b64encode(img_file.read()).decode()

# # === Streamlit UI ===
# st.set_page_config(page_title="Поиск животного", layout="wide")
# st.title("🔎 Поиск пропавшего животного по фото")

# # Сессия
# for key in ["map", "coords", "confirmed_coords", "image", "center_on", "nearby", "faraway", "others"]:
#     if key not in st.session_state:
#         st.session_state[key] = None

# st.markdown("""
#     <style>
#     .scroll-container {
#         max-height: 600px;
#         overflow-y: auto;
#         padding-right: 10px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# uploaded_files = st.file_uploader(
#     "📸 Загрузите до 10 фотографий животного",
#     type=["jpg", "jpeg", "png"],
#     accept_multiple_files=True
# )

# if uploaded_files:
#     uploaded_files = uploaded_files[:10]
#     images = [Image.open(file) for file in uploaded_files]
#     st.session_state['image'] = images

#     for i, img in enumerate(images):
#         st.image(img, caption=f"Изображение {i+1}", width=250)

#     st.markdown("### 🗺️ Отметьте на карте точку, где вы в последний раз видели животное.")
#     default_loc = [55.75, 37.62]
#     pick_map = folium.Map(location=default_loc, zoom_start=12)
#     pick_map.add_child(folium.LatLngPopup())
#     pick_data = st_folium(pick_map, width=700, height=500)

#     if pick_data and pick_data.get("last_clicked"):
#         lat = pick_data["last_clicked"]["lat"]
#         lon = pick_data["last_clicked"]["lng"]
#         st.session_state['coords'] = (lat, lon)
#         # st.info(f"📌 Координаты выбраны: {lat:.5f}, {lon:.5f}")

#         # if st.button("✅ Подтвердить координаты"):
#         #     st.session_state['confirmed_coords'] = (lat, lon)
#         #     st.success("Координаты подтверждены. Можно запускать поиск.")
#         col_map, col_button = st.columns([4, 1])
#         with col_map:
#             pick_data = st_folium(pick_map, width=700, height=500)

#         with col_button:
#             if pick_data and pick_data.get("last_clicked"):
#                 lat = pick_data["last_clicked"]["lat"]
#                 lon = pick_data["last_clicked"]["lng"]
#                 st.session_state['coords'] = (lat, lon)

#                 if st.button("📍 Подтвердить точку"):
#                     st.session_state['confirmed_coords'] = (lat, lon)
#                     st.success("Точка подтверждена. Можно запускать поиск.")

#                     # Ставим маркер пропажи и показываем всех животных (оранжевым)
#                     map_preview = folium.Map(location=[lat, lon], zoom_start=13)
#                     folium.Marker([lat, lon], popup="📍 Место пропажи", icon=folium.Icon(color="red")).add_to(map_preview)

#                     with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
#                         all_locations = json.load(f)

#                     for animal_id, coords in all_locations.items():
#                         dist = geodesic((lat, lon), (coords["lat"], coords["lon"])).km
#                         img_path = get_first_image(animal_id)
#                         if img_path:
#                             sim = get_embedding(Image.open(img_path))
#                             for user_img in st.session_state['image']:
#                                 user_emb = get_embedding(user_img)
#                                 s = cosine_similarity(user_emb.unsqueeze(0), sim.unsqueeze(0)).item()
#                                 if s > 0.6:
#                                     encoded = image_to_base64(img_path)
#                                     html = f"""
#                                     <b>ID:</b> {animal_id}<br>
#                                     <b>Пороговое сходство:</b> {s:.3f}<br>
#                                     <img src='data:image/jpeg;base64,{encoded}' width='150'>
#                                     """
#                                     folium.Marker(
#                                         location=[coords["lat"], coords["lon"]],
#                                         popup=folium.Popup(html, max_width=250),
#                                         icon=folium.Icon(color="orange")
#                                     ).add_to(map_preview)

#                     st.session_state['map'] = map_preview

# if st.session_state['confirmed_coords'] and st.session_state['image']:
#     if st.button("🔍 Найти похожих"):
#         with st.spinner("🔍 Идёт поиск..."):
#             lat, lon = st.session_state['confirmed_coords']
#             images = st.session_state['image']
#             nearby, faraway, others = find_similar_animals(images, lat, lon)
#             st.session_state['nearby'] = nearby
#             st.session_state['faraway'] = faraway
#             st.session_state['others'] = others
#             st.session_state['center_on'] = None

#             all_animals = nearby + faraway + others
#             st.session_state['results'] = sorted(all_animals, key=lambda x: x[2], reverse=True)

#             def make_map(lat, lon, center_on=None):
#                 m = folium.Map(location=center_on or [lat, lon], zoom_start=15 if center_on else 13)
#                 folium.Marker([lat, lon], popup="📍 Место пропажи", icon=folium.Icon(color="red")).add_to(m)
#                 for group, color in zip([nearby, faraway, others], ["green", "pink", "orange"]):
#                     for aid, coords, sim, dist, img_path in group:
#                         if img_path:
#                             encoded = image_to_base64(img_path)
#                             html = f"""
#                             <b>ID:</b> {aid}<br>
#                             <b>Сходство:</b> {sim:.3f}<br>
#                             <b>Расстояние:</b> {dist:.2f} км<br>
#                             <img src='data:image/jpeg;base64,{encoded}' width='150'>
#                             """
#                             folium.Marker(
#                                 location=[coords["lat"], coords["lon"]],
#                                 popup=folium.Popup(html, max_width=250),
#                                 icon=folium.Icon(color=color)
#                             ).add_to(m)
#                 return m

#             result_map = make_map(lat, lon)
#             st.session_state['map'] = result_map
#             st.success("✅ Поиск завершён")

# if st.session_state['map']:
#     st.subheader("🗺️ Найденные животные")
#     col1, col2 = st.columns([2, 1])

#     with col2:
#         st.markdown("### 📋 Карточки животных")
#         st.markdown('<div class="scroll-container">', unsafe_allow_html=True)

#         def display_card(aid, coords, sim, dist, img_path):
#             st.image(img_path, width=200)
#             st.markdown(f"**ID:** `{aid}`")
#             st.markdown(f"Сходство: `{sim:.3f}`")
#             st.markdown(f"Расстояние: `{dist:.2f} км`")
#             if st.button(f"📍 Центрировать на {aid}", key=f"center_{aid}"):
#                 st.session_state['center_on'] = (coords["lat"], coords["lon"])

#         # for animal in st.session_state.get("results", []):
#         #     display_card(*animal)
#         if "page" not in st.session_state:
#             st.session_state["page"] = 0

#         results = st.session_state.get("results", [])
#         start = st.session_state["page"] * 10
#         end = start + 10
#         shown_results = results[start:end]

#         for animal in shown_results:
#             display_card(*animal)

#         if end < len(results):
#             if st.button("📦 Показать ещё"):
#                 st.session_state["page"] += 1

#         st.markdown('</div>', unsafe_allow_html=True)

#     with col1:
#         lat, lon = st.session_state['confirmed_coords']
#         center = st.session_state.get("center_on")
#         updated_map = folium.Map(location=center or [lat, lon], zoom_start=15 if center else 13)
#         updated_map.add_child(folium.LatLngPopup())
#         for group, color in zip([st.session_state['nearby'], st.session_state['faraway'], st.session_state['others']], ["green", "pink", "orange"]):
#             for aid, coords, sim, dist, img_path in group:
#                 if img_path:
#                     encoded = image_to_base64(img_path)
#                     html = f"""
#                     <b>ID:</b> {aid}<br>
#                     <b>Сходство:</b> {sim:.3f}<br>
#                     <b>Расстояние:</b> {dist:.2f} км<br>
#                     <img src='data:image/jpeg;base64,{encoded}' width='150'>
#                     """
#                     folium.Marker(
#                         location=[coords["lat"], coords["lon"]],
#                         popup=folium.Popup(html, max_width=250),
#                         icon=folium.Icon(color=color)
#                     ).add_to(updated_map)

#         folium.Marker(
#             location=[lat, lon],
#             popup="📍 Место пропажи",
#             icon=folium.Icon(color="red")
#         ).add_to(updated_map)

#         st_folium(updated_map, width=700, height=600)
# === Импорты ===
import os
import json
import torch
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.functional import cosine_similarity
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from base64 import b64encode

# === Параметры ===
DATASET_DIR = "/Users/lolovolkova/Desktop/pet_app_airi/id_pets"
LOCATIONS_PATH = "/Users/lolovolkova/Desktop/pet_app_airi/animal_locations.json"
TOP_N = 10

# === Устройство ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Модель и трансформации ===
@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform

model, transform = load_model()

def get_embedding(image: Image.Image):
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.squeeze(0)

def get_first_image(animal_id):
    folder = os.path.join(DATASET_DIR, animal_id)
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(folder, file)
    return None

def image_to_base64(path):
    with open(path, "rb") as img_file:
        return b64encode(img_file.read()).decode()

def find_similar_animals(uploaded_images, lat, lon):
    similarity_by_animal = {}
    target_embeddings = [get_embedding(img) for img in uploaded_images]

    for animal_id in os.listdir(DATASET_DIR):
        animal_path = os.path.join(DATASET_DIR, animal_id)
        if not os.path.isdir(animal_path):
            continue

        max_sim = -1
        for fname in os.listdir(animal_path):
            img_path = os.path.join(animal_path, fname)
            try:
                img = Image.open(img_path)
                emb = get_embedding(img)
                for target_emb in target_embeddings:
                    sim = cosine_similarity(target_emb.unsqueeze(0), emb.unsqueeze(0)).item()
                    if sim > max_sim:
                        max_sim = sim
            except:
                continue

        if max_sim > 0.6:
            similarity_by_animal[animal_id] = max_sim

    with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
        locations = json.load(f)

    top_animals = sorted(similarity_by_animal.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    nearby, faraway, others = [], [], []

    for animal_id, sim in top_animals:
        if animal_id in locations:
            coords = locations[animal_id]
            dist = geodesic((lat, lon), (coords["lat"], coords["lon"])).km
            first_img = get_first_image(animal_id)
            info = (animal_id, coords, sim, dist, first_img)
            if dist <= 1.0:
                nearby.append(info)
            else:
                faraway.append(info)

    others_ids = set(similarity_by_animal.keys()) - set([a[0] for a in top_animals])
    for animal_id in others_ids:
        if animal_id in locations:
            coords = locations[animal_id]
            dist = geodesic((lat, lon), (coords["lat"], coords["lon"])).km
            first_img = get_first_image(animal_id)
            others.append((animal_id, coords, similarity_by_animal[animal_id], dist, first_img))

    return nearby, faraway, others

# === Streamlit UI ===
st.set_page_config(page_title="Поиск животного", layout="wide")
st.title("🔎 Поиск пропавшего животного по фото")

for key in ["map", "coords", "confirmed_coords", "image", "center_on", "nearby", "faraway", "others", "results", "page"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "page" else 0

st.markdown("""
    <style>
    .scroll-container {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("📸 Загрузите до 10 фотографий животного", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    uploaded_files = uploaded_files[:10]
    images = [Image.open(file) for file in uploaded_files]
    st.session_state['image'] = images

    for i, img in enumerate(images):
        st.image(img, caption=f"Изображение {i+1}", width=250)

    st.markdown("### 🗺️ Отметьте на карте точку, где вы в последний раз видели животное.")
    default_loc = [55.75, 37.62]
    col_map, col_button = st.columns([4, 1])

    with col_map:
        pick_map = folium.Map(location=default_loc, zoom_start=12)
        pick_map.add_child(folium.LatLngPopup())
        pick_data = st_folium(pick_map, width=700, height=500)

    with col_button:
        if pick_data and pick_data.get("last_clicked"):
            lat = pick_data["last_clicked"]["lat"]
            lon = pick_data["last_clicked"]["lng"]
            st.session_state['coords'] = (lat, lon)

            if st.button("📍 Подтвердить точку"):
                st.session_state['confirmed_coords'] = (lat, lon)

                preview_map = folium.Map(location=[lat, lon], zoom_start=13)
                folium.Marker([lat, lon], popup="📍 Место пропажи", icon=folium.Icon(color="red")).add_to(preview_map)

                with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
                    all_locations = json.load(f)

                for animal_id, coords in all_locations.items():
                    img_path = get_first_image(animal_id)
                    if img_path:
                        try:
                            sim = get_embedding(Image.open(img_path))
                            for user_img in images:
                                user_emb = get_embedding(user_img)
                                s = cosine_similarity(user_emb.unsqueeze(0), sim.unsqueeze(0)).item()
                                if s > 0.6:
                                    encoded = image_to_base64(img_path)
                                    html = f"""
                                    <b>ID:</b> {animal_id}<br>
                                    <img src='data:image/jpeg;base64,{encoded}' width='150'>
                                    """
                                    folium.Marker(
                                        location=[coords["lat"], coords["lon"]],
                                        popup=folium.Popup(html, max_width=250),
                                        icon=folium.Icon(color="orange")
                                    ).add_to(preview_map)
                                    break
                        except:
                            continue

                st.session_state['map'] = preview_map
                st_folium(preview_map, width=700, height=600)

if st.session_state['confirmed_coords'] and st.session_state['image']:
    if st.button("🔍 Найти похожих"):
        with st.spinner("🔍 Идёт поиск..."):
            lat, lon = st.session_state['confirmed_coords']
            images = st.session_state['image']
            nearby, faraway, others = find_similar_animals(images, lat, lon)
            st.session_state.update({"nearby": nearby, "faraway": faraway, "others": others, "center_on": None})
            all_animals = nearby + faraway + others
            st.session_state['results'] = sorted(all_animals, key=lambda x: x[2], reverse=True)

if st.session_state['results']:
    st.subheader("🗺️ Найденные животные")
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### 📋 Карточки животных")
        st.markdown('<div class="scroll-container">', unsafe_allow_html=True)

        def display_card(aid, coords, sim, dist, img_path):
            st.image(img_path, width=200)
            st.markdown(f"**ID:** `{aid}`")
            st.markdown(f"Сходство: `{sim:.3f}`")
            st.markdown(f"Расстояние: `{dist:.2f} км`")
            if st.button(f"📍 Центрировать на {aid}", key=f"center_{aid}"):
                st.session_state['center_on'] = (coords["lat"], coords["lon"])

        start = st.session_state["page"] * 10
        end = start + 10
        for animal in st.session_state['results'][start:end]:
            display_card(*animal)

        if end < len(st.session_state['results']):
            if st.button("📦 Показать ещё"):
                st.session_state["page"] += 1

        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        lat, lon = st.session_state['confirmed_coords']
        center = st.session_state.get("center_on")
        updated_map = folium.Map(location=center or [lat, lon], zoom_start=15 if center else 13)
        updated_map.add_child(folium.LatLngPopup())

        for group, color in zip([st.session_state['nearby'], st.session_state['faraway'], st.session_state['others']], ["green", "pink", "orange"]):
            for aid, coords, sim, dist, img_path in group:
                if img_path:
                    encoded = image_to_base64(img_path)
                    html = f"""
                    <b>ID:</b> {aid}<br>
                    <b>Сходство:</b> {sim:.3f}<br>
                    <b>Расстояние:</b> {dist:.2f} км<br>
                    <img src='data:image/jpeg;base64,{encoded}' width='150'>
                    """
                    folium.Marker(
                        location=[coords["lat"], coords["lon"]],
                        popup=folium.Popup(html, max_width=250),
                        icon=folium.Icon(color=color)
                    ).add_to(updated_map)

        folium.Marker(
            location=[lat, lon],
            popup="📍 Место пропажи",
            icon=folium.Icon(color="red")
        ).add_to(updated_map)

        st_folium(updated_map, width=700, height=600)