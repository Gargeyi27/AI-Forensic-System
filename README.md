#  Forensic AI Toolkit 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from google.colab import files

# ======================================
# 🔍 Fingerprint Feature Encoder (CNN)
# ======================================
class FingerprintFeatureEncoder:
    def __init__(self):
        base_net = ResNet50(weights='imagenet', include_top=False)
        gap_layer = GlobalAveragePooling2D()(base_net.output)
        embedding = Dense(128, activation='relu')(gap_layer)
        self.extractor = Model(inputs=base_net.input, outputs=embedding)
        print("[SYSTEM] Feature extractor initialized successfully.")

    def preprocess_image(self, img_file):
        img = load_img(img_file, target_size=(224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        return tf.keras.applications.resnet50.preprocess_input(arr)

    def get_embedding(self, img_file):
        arr = self.preprocess_image(img_file)
        vector = self.extractor.predict(arr)[0]
        return vector / np.linalg.norm(vector)

    def compare_fingerprints(self, query_file, db_files):
        query_vector = self.get_embedding(query_file).reshape(1, -1)
        scores = []
        for sid, file in db_files.items():
            db_vector = self.get_embedding(file).reshape(1, -1)
            similarity = cos_sim(query_vector, db_vector)[0][0]
            scores.append((sid, similarity))
        return sorted(scores, key=lambda x: x[1], reverse=True)

# ======================================
#  Blood Pattern Recognition
# ======================================
def identify_blood_patterns(scene_file):
    image = cv2.imread(scene_file)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv_image, (0,50,50), (10,255,255))
    red_mask2 = cv2.inRange(hsv_image, (170,50,50), (180,255,255))
    combined_mask = cv2.bitwise_or(red_mask1, red_mask2)
    output = cv2.bitwise_and(image, image, mask=combined_mask)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Original Scene")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1,2,2)
    plt.title("Detected Blood Patterns")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.show()

# ======================================
#  Projectile Trajectory Simulation
# ======================================
def calculate_projectile():
    print("\n[INPUT] Provide firearm parameters")
    try:
        velocity = float(input("Muzzle Velocity (m/s): "))
        theta = float(input("Firing Angle (degrees): "))
    except ValueError:
        print("[ERROR] Invalid values, defaulting to preset parameters.")
        velocity, theta = 700, 40

    gravity = 9.81
    t_max = (2 * velocity * np.sin(np.radians(theta))) / gravity
    t_points = np.linspace(0, t_max, 500)

    x_coords = velocity * np.cos(np.radians(theta)) * t_points
    y_coords = velocity * np.sin(np.radians(theta)) * t_points - 0.5 * gravity * t_points**2

    plt.plot(x_coords, y_coords, color='purple')
    plt.title("Projectile Path Simulation")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.grid(True)
    plt.show()

# ======================================
#  Crime Scene 3D Mapping
# ======================================
def map_crime_scene():
    evidence_points = np.random.uniform(0, 50, size=(60,3))
    clusterer = KMeans(n_clusters=5, random_state=99)
    cluster_labels = clusterer.fit_predict(evidence_points)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for cluster_id in np.unique(cluster_labels):
        ax.scatter(evidence_points[cluster_labels==cluster_id,0],
                   evidence_points[cluster_labels==cluster_id,1],
                   evidence_points[cluster_labels==cluster_id,2],
                   label=f"Cluster-{cluster_id+1}")
    ax.set_title("3D Evidence Mapping")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    plt.show()

# ======================================
#  Integrated Demo
# ======================================
def launch_demo():
    print("\n Forensic AI System: Interactive Demo")

    # Fingerprint Matching
    print("\n Upload Reference Fingerprint")
    uploaded_query = files.upload()
    query_img = list(uploaded_query.keys())[0]

    print("\n Upload Database Fingerprints")
    uploaded_db = files.upload()
    db_images = {f"Suspect-{i+1:02}": img for i, img in enumerate(uploaded_db.keys())}

    encoder = FingerprintFeatureEncoder()
    result_scores = encoder.compare_fingerprints(query_img, db_images)
    print("\n Similarity Results:")
    for sid, score in result_scores:
        print(f"{sid}: {score*100:.2f}%")

    # Blood Detection
    print("\n Upload Crime Scene Photo for Blood Detection")
    uploaded_scene = files.upload()
    scene_img = list(uploaded_scene.keys())[0]
    identify_blood_patterns(scene_img)

    # Ballistics Simulation
    print("\n Simulating Projectile Motion")
    calculate_projectile()

    # 3D Scene Mapping
    print("\n Reconstructing 3D Crime Scene")
    map_crime_scene()

if __name__ == "__main__":
    launch_demo()
