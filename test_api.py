import requests

BASE_URL = "http://localhost:8000"

# 1 Health Check
response = requests.get(f"{BASE_URL}/")
assert response.status_code == 200
print("✅ Health Check Passed:", response.json())


# 2️ Classes Endpoint
response = requests.get(f"{BASE_URL}/classes")
assert response.status_code == 200
classes = response.json()
print("✅ Total Classes:", len(classes))


# 3️ Image Detection
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/detect/image",
        files={"file": ("test.jpg", f, "image/jpeg")},
        params={"confidence": 0.4, "classes": ""}
    )

assert response.status_code == 200
print("✅ Image Detection Passed")
print("Detections:", response.headers.get("X-Detections"))


# 4️ Class Filter (Only Person)
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/detect/image",
        files={"file": ("test.jpg", f, "image/jpeg")},
        params={"confidence": 0.4, "classes": "person"}
    )

assert response.status_code == 200
print("✅ Class Filter Passed (Person Only)")


# 5️ Invalid Image Test
response = requests.post(
    f"{BASE_URL}/detect/image",
    files={"file": ("bad.txt", b"not an image", "text/plain")}
)

assert response.status_code == 400
print("✅ Invalid Image Test Passed")