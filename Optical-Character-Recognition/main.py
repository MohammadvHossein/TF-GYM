import easyocr
import cv2
import matplotlib.pyplot as plt
from textblob import TextBlob
import warnings
import os

warnings.filterwarnings("ignore")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "pos"
    elif analysis.sentiment.polarity < 0:
        return "neg"
    else:
        return "neutral"

def process_image(image_path, save_image=False, show_image=True, save_text=False , sensitivity=False):

    image = cv2.imread(image_path)

    result = reader.readtext(image)

    
    if save_text:
        full_text = ""
        if sensitivity == False:
            for (bbox, text, prob) in result:
                sentiment = analyze_sentiment(text)
                full_text += f"{text} - Sentiment: {sentiment}\n"
        else:
            full_text = ""
            current_sentence = ""
            for (bbox, text, prob) in result:
            # Append text to the current sentence
                current_sentence += text + " "
                # sensitivity: check for '.', '!', or '?'
                if text.endswith(('.', '!', '?')):
                    sentiment = analyze_sentiment(current_sentence.strip())
                    full_text += f"{current_sentence.strip()} - Sentiment: {sentiment}\n"
                    current_sentence = ""  # Reset for the next sentence
            if current_sentence:
                sentiment = analyze_sentiment(current_sentence.strip())
                full_text += f"{current_sentence.strip()} - Sentiment: {sentiment}\n"


        # Save the full text to a file
        with open('extracted_text.txt', 'w') as f:
            f.write(full_text)

    # Optionally, display the image with detected text
    if show_image:
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            sentiment = analyze_sentiment(text)
            cv2.putText(image, sentiment, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the image with detected text if required
    if save_image:
        plt.imsave('result.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Display the image if required
    if show_image:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        

def process_in_folder(folder_path, save_image=False, show_image=False, sensitivity=False):
    # Create a file to store the results
    result_file = open('results.txt', 'w')

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG'):
            image_path = os.path.join(folder_path, filename)
            result = reader.readtext(image_path)

            # Extract text and sentiment from the image
            full_text = ""
            if sensitivity == False:
                for (bbox, text, prob) in result:
                    sentiment = analyze_sentiment(text)
                    full_text += f"{filename} - {text} - Sentiment: {sentiment}\n"
            else:
                current_sentence = ""
                for (bbox, text, prob) in result:
                    # Append text to the current sentence
                    current_sentence += text + " "
                    # sensitivity: check for '.', '!', or '?'
                    if text.endswith(('.', '!', '?')):
                        sentiment = analyze_sentiment(current_sentence.strip())
                        full_text += f"{filename} - {current_sentence.strip()} - Sentiment: {sentiment}\n"
                        current_sentence = ""  # Reset for the next sentence
                if current_sentence:
                    sentiment = analyze_sentiment(current_sentence.strip())
                    full_text += f"{filename} - {current_sentence.strip()} - Sentiment: {sentiment}\n"

            # Save the extracted text to the result file
            result_file.write(full_text)

            # Optionally, save and display the processed images
            if save_image:
                plt.imsave(os.path.join(folder_path, f"result_{filename}"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if show_image:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.show()

    # Close the result file
    result_file.close()