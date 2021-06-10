import face_recognition
from pathlib import Path
from PIL import Image

known_image = face_recognition.load_image_file("criminal.jpg")

known_image_encoding = face_recognition.face_encodings(known_image)[0]

best_face_distance = 1.0
best_face_image = None

for image_path in Path("CCTVstream").glob("*.png"):

    unknown_image = face_recognition.load_image_file(image_path)

    face_encodings = face_recognition.face_encodings(unknown_image)

    face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]

    if face_distance < best_face_distance:
        best_face_distance = face_distance
        best_face_image = unknown_image

pil_image = Image.fromarray(best_face_image)
pil_image.show()
