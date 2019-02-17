import cv2
import tensorflow as tf

CATEGORIES = ["Actinic keratoses (Cancerous)", "Basal cell carsinoma (Cancerous)", "Benign keratosis (Not Cancerous)", "Dermatofibroma (Not Cancerous)", "Melanoma (Cancerous)", "Vascular lesions (Cancerous or Not Cancerous)"]

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("")

prediction = model.predict([prepare('')])

print(CATEGORIES[int(prediction[0][0])])