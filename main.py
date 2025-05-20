from flask import Flask, request, jsonify
from flask import render_template
import settings
import utils
import numpy as np
import cv2
import predictions as pred
import os
from flask import session
import json
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'document_scanner_app'
# Set session to be permanent with longer timeout
app.permanent_session_lifetime = timedelta(hours=1)

docscan = utils.DocumentScan()

# Initialize model variables as None
qwen_model = None
qwen_processor = None

@app.route('/',methods=['GET','POST'])
def scandoc():
    
    if request.method == 'POST':
        file = request.files['image_name']
        ocr_model = request.form.get('ocr_model', 'pytesseract')
        
        # Make session permanent and set OCR model
        session.permanent = True
        
        session['ocr_model'] = ocr_model
        
        upload_image_path = utils.save_upload_image(file)
        print('Image saved in = ',upload_image_path)
        # predict the coordination of the document
        four_points, size = docscan.document_scanner(upload_image_path)
        print(four_points,size)
        if four_points is None:
            message ='UNABLE TO LOCATE THE COORDINATES OF DOCUMENT: points displayed are random'
            points = [
                {'x':10 , 'y': 10},
                {'x':120 , 'y': 10},
                {'x':120 , 'y': 120},
                {'x':10 , 'y': 120}
            ]
            return render_template('scanner.html',
                                   points=points,
                                   fileupload=True,
                                   ocr_model=ocr_model,
                                   message=message)
            
        else:
            points = utils.array_to_json_format(four_points)
            message ='Located the Cooridinates of Document using OpenCV'
            return render_template('scanner.html',
                                   points=points,
                                   fileupload=True,
                                   ocr_model=ocr_model,
                                   message=message)
        
        return render_template('scanner.html')
    
    # For GET requests, get the OCR model from session if available
    ocr_model = session.get('ocr_model', 'pytesseract')
    return render_template('scanner.html', ocr_model=ocr_model)


@app.route('/load_qwen_model', methods=['POST'])
def load_qwen_model():
    global qwen_model, qwen_processor
    
    try:
        if qwen_model is None or qwen_processor is None:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Define model path
            model_path = "./Qwen2-VL-2B-OCR-fp16"
            
            # Load processor
            print("Loading processor...")
            qwen_processor = AutoProcessor.from_pretrained(
                model_path, 
                size={"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
            )
            
            # Load model
            print("Loading model...")
            qwen_model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            return jsonify({"status": "success", "message": "Model loaded successfully"})
        else:
            return jsonify({"status": "success", "message": "Model already loaded"})
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to load model: {str(e)}"}), 500


@app.route('/transform',methods=['POST'])
def transform():
    try:
        # Keep the model selection from the session
        # The ocr_model value is already set in the scandoc route, 
        # but we'll ensure it's maintained for the prediction route
        if 'ocr_model' not in session:
            session['ocr_model'] = 'pytesseract'
            
        points = request.json['data']
        array = np.array(points)
        magic_color = docscan.calibrate_to_original_size(array)
        #utils.save_image(magic_color,'magic_color.jpg')
        filename =  'magic_color.jpg'
        magic_image_path = settings.join_path(settings.MEDIA_DIR,filename)
        cv2.imwrite(magic_image_path,magic_color)
        
        return 'sucess'
    except:
        return 'fail'
        
    
@app.route('/prediction')
def prediction():
    global qwen_model, qwen_processor
    
    # Get the selected OCR model from session
    ocr_model = session.get('ocr_model', 'pytesseract')
    
    # Ensure the session variable is maintained
    print(f"OCR model from session: {ocr_model}")
    
    if ocr_model == 'qwen2':
        # Check if Qwen model is loaded
        if qwen_model is None or qwen_processor is None:
            return render_template('predictions.html', 
                                  results={"ERROR": "Qwen2 model not loaded. Please go back and load the model first."})
        
        # Use Qwen2 model for entity extraction
        try:
            from PIL import Image
            import torch
            import json
            
            # For Qwen2, we can use the original uploaded image directly
            upload_image_path = settings.join_path(settings.MEDIA_DIR, 'upload.jpg')
            
            # Check if the image exists
            if not os.path.exists(upload_image_path):
                return render_template('predictions.html', 
                                      results={"ERROR": "Image file not found. Please upload an image first."})
            
            # Save a copy for display as bounding box image
            bb_filename = settings.join_path(settings.MEDIA_DIR, 'bounding_box.jpg')
            image = cv2.imread(upload_image_path)
            
            if image is None:
                return render_template('predictions.html', 
                                      results={"ERROR": "Failed to read image file. The file may be corrupted."})
            
            cv2.imwrite(bb_filename, image)
            
            # Load image with PIL for Qwen2
            try:
                pil_image = Image.open(upload_image_path).convert("RGB")
                
                # Resize image
                pil_image = pil_image.resize((640, 640))
            except Exception as e:
                return render_template('predictions.html', 
                                      results={"ERROR": f"Failed to process image with PIL: {str(e)}"})
            
            # Prompt for entity extraction
            prompt_text = """Extract NAME, ORG, DES, PHONE, EMAIL, WEB from the image. Respond as JSON. Only use visible info.
{
  "NAME": [],
  "ORG": [],
  "DES": [],
  "PHONE": [],
  "EMAIL": [],
  "WEB": []
}"""
            
            # Create conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            # Apply chat template and tokenize
            try:
                prompt = qwen_processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = qwen_processor(
                    text=[prompt],
                    images=[pil_image],
                    padding=True,
                    return_tensors="pt"
                ).to(qwen_model.device)
            except Exception as e:
                print(f"Error in processing inputs: {str(e)}")
                return render_template('predictions.html', 
                                      results={"ERROR": f"Failed to process model inputs: {str(e)}"})
            
            # Generate output
            try:
                with torch.no_grad():
                    output_ids = qwen_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        num_beams=1,
                        early_stopping=True
                    )
                
                # Decode output
                generated_text = qwen_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            except Exception as e:
                print(f"Error in model generation: {str(e)}")
                return render_template('predictions.html', 
                                      results={"ERROR": f"Error in AI model processing: {str(e)}"})
            
            # Handle empty or invalid responses
            if not generated_text or len(generated_text.strip()) == 0:
                return render_template('predictions.html', 
                                      results={"ERROR": "The AI model returned an empty response."})

            # Extract JSON as string from the response using the existing function
            try:
                json_str = pred.extract_json_response(generated_text)
                print("Extracted JSON string:", json_str)
                
                # Convert JSON string to dictionary
                if isinstance(json_str, str) and '{' in json_str:
                    # Parse the JSON string into a dictionary
                    results = json.loads(json_str)
                else:
                    # In case it's not valid JSON
                    results = {"Extracted Text (Qwen2-VL)": json_str}
                
                # Ensure all required keys are present
                for key in ["NAME", "ORG", "DES", "PHONE", "EMAIL", "WEB"]:
                    if key not in results:
                        results[key] = []
                
                print("Processed results:", results)
            except json.JSONDecodeError as e:
                # Fallback if JSON parsing fails
                print(f"JSON decode error: {str(e)}")
                results = {"ERROR": f"Failed to parse JSON response: {str(e)}",
                          "Raw Text": json_str}
            except Exception as e:
                print(f"Error processing JSON: {str(e)}")
                results = {"ERROR": f"Error processing model response: {str(e)}",
                          "Raw Text": json_str if 'json_str' in locals() else "No JSON extracted"}
            
            # Use the new template for Qwen2 results
            return render_template('qwen_prediction.html', results=results)
        except Exception as e:
            print(f"Unhandled exception in Qwen2 processing: {str(e)}")
            return render_template('predictions.html', 
                                  results={"ERROR": f"Error processing with Qwen2: {str(e)}"})
    else:
        try:
            # load the wrap image for Pytesseract/Spacy processing
            wrap_image_filepath = settings.join_path(settings.MEDIA_DIR,'magic_color.jpg')
            
            # Check if the wrapped image exists
            if not os.path.exists(wrap_image_filepath):
                return render_template('predictions.html', 
                                      results={"ERROR": "Wrapped image not found. Please process the document first."})
                
            image = cv2.imread(wrap_image_filepath)
            
            if image is None:
                return render_template('predictions.html', 
                                      results={"ERROR": "Failed to read wrapped image. The file may be corrupted."})
                
            # Use the original Pytesseract + SpaCy NER method
            image_bb, results = pred.getPredictions(image)
            
            bb_filename = settings.join_path(settings.MEDIA_DIR,'bounding_box.jpg') 
            cv2.imwrite(bb_filename, image_bb)
            
            # If results contain an ERROR key, it means the prediction failed
            if "ERROR" in results:
                print(f"Error in Pytesseract processing: {results['ERROR']}")
                
            return render_template('predictions.html', results=results)
        except Exception as e:
            print(f"Unhandled exception in Pytesseract processing: {str(e)}")
            return render_template('predictions.html', 
                                  results={"ERROR": f"Error in document processing: {str(e)}"})


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)