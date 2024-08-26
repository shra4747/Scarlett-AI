video_path = "/notebooks/videos/dcmp58.mp4"
output_path = "/notebooks/videos/output/dcmp58-ilt.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

bounding_box_annotator = sv.BoxAnnotator()

frame_count = 0
lime = LIME(iterations=1, alpha=1.5, rho=1.5, gamma=0.5, strategy=1)

previousRedBoxes = {}
previousBlueBoxes = {}


def sort_boxes_by_distance(previousBoxes, curr_centroid_x, curr_centroid_y):
    def calculate_distance(box):
        prev_centroid_x = (box["x1"] + box["x2"]) / 2
        prev_centroid_y = (box["y1"] + box["y2"]) / 2
        distance = sqrt((curr_centroid_x - prev_centroid_x) ** 2 + (curr_centroid_y - prev_centroid_y) ** 2)
        return distance

    sorted_boxes = sorted(previousBoxes.values(), key=calculate_distance)
    return sorted_boxes


with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
    while True:
        ret, frame = cap.read()
        
        
        if not ret:
            print("Finished processing all frames.")
            break

        if frame is None:
            print("Empty frame encountered.")
            continue

        # Convert the frame from BGR (OpenCV) to RGB (PIL)
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(image_pil)


        blue_boxes = []
        red_boxes = []

        red_team_numbers = original_red_team_numbers.copy()
        blue_team_numbers = original_blue_team_numbers.copy()
        
        # Separate boxes by team color
        for result in results:
            boxes = result.boxes
            boxes = sorted(boxes, key=lambda box: box.conf, reverse=True)
            ocr_count = 0
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                start_row = (y1 + y2) // 2
                bottom_half = frame[start_row:y2, x1:x2]

                # Calculate average RGB to determine the team color
                average_rgb = np.mean(bottom_half, axis=(0, 1))
                label = 'Blue' if average_rgb[0] > average_rgb[2] else 'Red'
                color = (255, 0, 0) if label == 'Blue' else (0, 0, 255)

                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                red_sorted_boxes = sort_boxes_by_distance(previousRedBoxes, centroid_x, centroid_y)
                blue_sorted_boxes = sort_boxes_by_distance(previousBlueBoxes, centroid_x, centroid_y)
                
                if len(red_sorted_boxes) != 0 and len(blue_sorted_boxes) != 0:
                    red_centroid_x = (red_sorted_boxes[0]['x1'] + red_sorted_boxes[0]['x2']) / 2
                    red_centroid_y = (red_sorted_boxes[0]['y1'] + red_sorted_boxes[0]['y2']) / 2
                    blue_centroid_x = (blue_sorted_boxes[0]['x1'] + blue_sorted_boxes[0]['x2']) / 2
                    blue_centroid_y = (blue_sorted_boxes[0]['y1'] + blue_sorted_boxes[0]['y2']) / 2

                    red_distance = sqrt((centroid_x - red_centroid_x) ** 2 + (centroid_y - red_centroid_y) ** 2)
                    blue_distance = sqrt((centroid_x - blue_centroid_x) ** 2 + (centroid_y - blue_centroid_y) ** 2)

                    if label == "Blue":
                        if red_distance < blue_distance:
                            label = "Red"
                            color = (0,0,255)
                        
                    if label == "Red":
                        if blue_distance < red_distance:
                            label = "Blue"
                            color = (255,0,0)


                # OCR: Recognize text in the bottom half of the box
                height, width, _ = bottom_half.shape
                upscaled_bottom_half = upscale_model.upsample(bottom_half)
                gray = cv2.cvtColor(upscaled_bottom_half, cv2.COLOR_BGR2GRAY)

                # Apply LIME enhancement
                lime.load(gray)
                enhanced_image = lime.enhance()

                result = ocr.ocr(np.array(enhanced_image), cls=True)
                ocr_result = ""
                try:
                    ocr_result = result[0][0][1][0]
                except:
                    enhanced_image = rotate(enhanced_image, -20)
                    result = ocr.ocr(np.array(enhanced_image), cls=True)
                    try:
                        ocr_result = result[0][0][1][0]
                    except:
                        enhanced_image = rotate(enhanced_image, 40)
                        result = ocr.ocr(np.array(enhanced_image), cls=True)
                        try:
                            ocr_result = result[0][0][1][0]
                        except:
                            ocr_result = ""
                

                detected_text = ocr_result.replace(" ", "") if ocr_result else ""
                if detected_text != "":
                    ocr_count += 1
                
                if label == 'Blue':
                    blue_boxes.append((x1, y1, x2, y2, detected_text, color))
                else:
                    red_boxes.append((x1, y1, x2, y2, detected_text, color))

        
        
    
        def assign_prev_boxes(alliance: str, box: dict):
            global previousBlueBoxes
            global previousRedBoxes

            if alliance.lower() == "red":
                previousRedBoxes[box['closest_robot_match']] = box
            else:
                previousBlueBoxes[box['closest_robot_match']] = box
            
        
        
        def assign_team_numbers(alliance:str, boxes:list):

            if alliance.lower() == "red":
                available_team_numbers = original_red_team_numbers.copy()
                # numbers = original_red_team_numbers.copy()
                previousBoxes = previousRedBoxes
            else:
                available_team_numbers = original_blue_team_numbers.copy()
                # numbers = original_blue_team_numbers.copy()
                previousBoxes = previousBlueBoxes

            for x1, y1, x2, y2, detected_text, color in boxes:          
                distances = {num: lev(detected_text, num, substitute_costs=substitute_costs) for num in available_team_numbers}
                sorted_distances = sorted(distances.items(), key=lambda item: item[1])
                
                closest_robot_match = sorted_distances[0][0]
                closest_robot_match_distance = sorted_distances[0][1]

                curr_centroid_x = (x1 + x2) / 2
                curr_centroid_y = (y1 + y2) / 2


                curr_robot_dict = {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "detected_text": detected_text,
                            "color": color,
                            "closest_robot_match": closest_robot_match, 
                            "closest_robot_match_distance": closest_robot_match_distance
                        }
                
                if len(previousBoxes) != 3:
                    assign_prev_boxes(alliance, curr_robot_dict)
                else:
                    sorted_boxes = sort_boxes_by_distance(previousBoxes, curr_centroid_x, curr_centroid_y)
                    closest_prev_box_by_distance = sorted_boxes[0]
                    
                    if closest_prev_box_by_distance["closest_robot_match_distance"] < closest_robot_match_distance:
                        curr_robot_dict = {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "detected_text": closest_prev_box_by_distance["detected_text"],
                            "color": closest_prev_box_by_distance["color"],
                            "closest_robot_match": closest_prev_box_by_distance["closest_robot_match"], 
                            "closest_robot_match_distance": closest_prev_box_by_distance["closest_robot_match_distance"]
                        }
                        assign_prev_boxes(alliance, curr_robot_dict)
                    else:
                        assign_prev_boxes(alliance, curr_robot_dict)

                cv2.rectangle(frame, (x1, y1), (x2, y2), curr_robot_dict['color'], 2)
                cv2.putText(frame, f"{curr_robot_dict['closest_robot_match']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, curr_robot_dict['color'], 2)
        
        # Assign team numbers to Blue and Red boxes
        assign_team_numbers(alliance="red", boxes=red_boxes)
        assign_team_numbers(alliance="blue", boxes=blue_boxes)
        # Write the frame with annotations to the output video
        out.write(frame)
        frame_count += 1
        pbar.update(1)

    # break
        
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved to {output_path}")