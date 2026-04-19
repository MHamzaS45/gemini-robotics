import os
from google import genai
from google.genai import types

#1 Fetch the API key and initialize our environment
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

#2 Upload the image to the Gemini API
image_path = "image.JPG"
my_image= client.files.upload(file = image_path)

#3: Add a prompt
prompt = """
Locate the bee plushie from the image and give me the bounding box coordinates in JSON format labels should be 
bounding_box: in the format [ymin, xmin, ymax, xmax] , and label: with the object name. 
Do not use any markdown, just the raw JSON
"""

#4: Call the gemini robotics model
image_response = client.models.generate_content(
    model="gemini-robotics-er-1.5-preview",
    contents=[
        my_image,
        prompt
    ],
    config = types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
)


print(image_response.text)
data_json = image_response.text

#5 Parse the JSON response
import json 
from PIL import Image, ImageDraw

data = json.loads(data_json)
box = data[0]['bounding_box']                   #  print the coordinates of the bounding box

print(box)

#6: Draw the bounding box on the image
image = Image.open(image_path)
w, h = image.size                                # Need the size to draw the bounding box correctly
draw = ImageDraw.Draw(image)                   # draw the bounding box on the image

# Convert normalized coordinates to pixel values
ymin, xmin, ymax, xmax = box  
draw.rectangle(
    [xmin / 1000 * w, ymin / 1000 * h, xmax / 1000 * w, ymax / 1000 * h],      # 1000 is used to convert the normalized coordinates to pixel values
    outline="red",
    width=3
)
