from openai import OpenAI
import os
import base64
import cv2

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def capture_image_from_webcam(image_path: str):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(image_path, frame)
        print(f"Image saved to {image_path}")
    else:
        print("Error: Could not capture image")

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path: str):
    # Open the local image file in binary mode
    with open(image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract text from this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500,
    )

    return response.choices[0].message.content

def save_to_txt(content: str, txt_filename: str):
    with open(txt_filename, 'w') as file:
        file.write(content)
    print(f"Text data saved to {txt_filename}")

image_path = 'webcam_image.png'

capture_image_from_webcam(image_path)

# Process the captured image and get the response
text_content = process_image(image_path)

# Save the recognized text to a .txt file
txt_filename = 'recognized_text.txt'
save_to_txt(text_content, txt_filename)
