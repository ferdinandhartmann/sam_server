import requests

url = "http://<SERVER_IP>:8000/reconstruct"

with open("mouse.jpg", "rb") as f:
    r = requests.post(url, files={"image": f})

print(r.json())


# response:
# {
#   "job_id": "...",
#   "obj": "/.../object.obj",
#   "stl": "/.../object.stl",
#   "png": "/.../preview.png"
# }
