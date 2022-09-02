"""
*
*
*
*
*
Inference code to test a TFLite Classification Model.

AUTHORS: Sencer YÜCEL & Behiç KILINÇKAYA

https://www.linkedin.com/in/yucelsencer/
https://www.linkedin.com/in/behic-kilinckaya/
*
*
*
*
*
"""

import gc
import image
import math
import os
import pyb
import sensor
import tf
import time
import ujson
import uos

PATH_TO_JSON = "PATH_TO_ANNOTATION_JSON_FILE"
PATH_TO_DATASET = "PATH_TO_YOUR_DATASET"
PATH_TO_CROPPED_PHOTOS_TO_SAVE = "PATH_TO_CROPPED_PHOTOS_TO_SAVE"
PATH_TO_TFLITE_MODEL = "PATH_TO_MODEL.tflite"
PATH_TO_SAVE_INFERENCE_RESULTS = "INFERENCE_RESULTS.txt"
CROP_COUNT = 16


# Tests the negatives.
def test_negative(detected_coords, json_file_name):
    true_negatives = 0
    false_negatives = 0
    condition_counter = 0
    detected_x1 = int(detected_coords[0])
    detected_y1 = int(detected_coords[1])
    detected_x2 = int(detected_coords[2])
    detected_y2 = int(detected_coords[3])

    json_path = "{}/{}".format(PATH_TO_JSON, json_file_name)
    json_file = ujson.load(open(json_path))
    for j in range(len(json_file["detections"])):
        if json_file["detections"][j]["label"] == "person":
            condition_counter = 1
            if json_file["detections"][j]["perc"] > 0.20:
                true_x1 = (json_file["detections"][j]["bbox"][0][0][0])
                true_y1 = (json_file["detections"][j]["bbox"][0][0][1])
                true_x2 = (json_file["detections"][j]["bbox"][0][1][0])
                true_y2 = (json_file["detections"][j]["bbox"][0][2][1])

                true_bbox_area = (true_y2 - true_y1) * (true_x2 - true_x1)
                detected_frame_area = (detected_y2 - detected_y1) * (detected_x2 - detected_x1)

                if true_bbox_area >= detected_frame_area:
                    # top left
                    if true_x1 <= detected_x2 <= true_x2 and true_y2 >= detected_y2 >= true_y1:
                        # left
                        if true_x2 >= detected_x1 >= true_x1 and true_y1 <= detected_y2 <= true_y2:
                            intersection_area = (detected_y2 - detected_y1) * (detected_x2 - true_x1)
                        # top
                        elif true_x2 >= detected_x1 >= true_x1 and true_y1 <= detected_y2 <= true_y2:
                            intersection_area = (detected_y2 - true_y1) * (detected_x2 - detected_x1)
                        # top left
                        else:
                            intersection_area = (detected_x2 - true_x1) * (detected_y2 - true_y1)

                        if intersection_area / true_bbox_area > 0.10:
                            false_negatives += 1
                        else:
                            true_negatives += 1

                    # top right
                    elif true_x2 >= detected_x1 >= true_x1 and true_y1 <= detected_y2 <= true_y2:
                        # right
                        if true_x1 <= detected_x1 <= true_x2 and true_y2 >= detected_y1 >= true_y1:
                            intersection_area = (detected_y2 - detected_y1) * (true_x2 - detected_x1)
                        else:
                            intersection_area = (true_x2 - detected_x1) * (detected_y2 - true_y1)

                        if intersection_area / true_bbox_area > 0.10:
                            false_negatives += 1
                        else:
                            true_negatives += 1

                    # bottom left
                    elif true_x1 <= detected_x2 <= true_x2 and true_y2 >= detected_y1 >= true_y1:
                        # bottom
                        if true_x1 <= detected_x1 <= true_x2 and true_y2 >= detected_y1 >= true_y1:
                            intersection_area = (detected_x2 - detected_x1) * (true_y2 - detected_y1)
                        else:
                            intersection_area = (detected_x2 - true_x1) * (true_y2 - detected_y1)

                        if intersection_area / true_bbox_area > 0.10:
                            false_negatives += 1
                        else:
                            true_negatives += 1

                    # bottom right
                    elif true_x1 <= detected_x1 <= true_x2 and true_y2 >= detected_y1 >= true_y1:
                        intersection_area = (true_x2 - detected_x1) * (true_y2 - detected_y1)
                        if intersection_area / true_bbox_area > 0.10:
                            false_negatives += 1

                    else:
                        true_negatives += 1

                # Frame > Bounding box
                else:
                    # top left
                    if detected_x2 >= true_x2 >= detected_x1 and detected_y2 >= true_y2 >= detected_y1:
                        # left
                        if detected_x2 >= true_x2 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                            intersection_area = (true_y2 - true_y1) * (true_x2 - detected_x1)
                        # top
                        elif detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y2 >= detected_y1:
                            intersection_area = (true_x2 - true_x1) * (true_y2 - detected_y1)
                        # top left
                        else:
                            intersection_area = (true_y2 - detected_y1) * (true_x2 - detected_x1)

                        if intersection_area / true_bbox_area > 0.40:
                            false_negatives += 1
                        else:
                            true_negatives += 1

                    # top right
                    elif detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y2 >= detected_y1:
                        # right
                        if detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                            intersection_area = (true_y2 - true_y1) * (detected_x2 - true_x1)
                        # top right
                        else:
                            intersection_area = (detected_x2 - true_x1) * (true_y2 - detected_y1)

                        if intersection_area / true_bbox_area > 0.40:
                            false_negatives += 1
                        else:
                            true_negatives += 1

                    # bottom left
                    elif detected_x2 >= true_x2 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                        # bottom
                        if detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                            intersection_area = (true_x2 - true_x1) * (detected_y2 - true_y1)
                        # bottom left
                        else:
                            intersection_area = (detected_y2 - true_y1) * (true_x2 - detected_x1)

                        if intersection_area / true_bbox_area > 0.40:
                            false_negatives += 1
                        else:
                            true_negatives += 1

                    # bottom right
                    elif detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                        intersection_area = (detected_x2 - true_x1) * (detected_y2 - true_y1)
                        if intersection_area / true_bbox_area > 0.40:
                            false_negatives += 1
                        else:
                            true_negatives += 1

                    else:
                        true_negatives += 1

    if condition_counter == 0:
        true_negatives += 1

    return [true_negatives, false_negatives]


# Tests the positives.
def test_positive(detected_coords, json_file_name):
    true_positives = 0
    false_positives = 0
    condition_counter = 0
    detected_x1 = int(detected_coords[0])
    detected_y1 = int(detected_coords[1])
    detected_x2 = int(detected_coords[2])
    detected_y2 = int(detected_coords[3])

    json_path = "{}/{}".format(PATH_TO_JSON, json_file_name)
    json_file = ujson.load(open(json_path))
    for j in range(len(json_file["detections"])):
        if json_file["detections"][j]["label"] == "person":
            condition_counter = 1
            if json_file["detections"][j]["perc"] > 0.20:
                true_x1 = (json_file["detections"][j]["bbox"][0][0][0])
                true_y1 = (json_file["detections"][j]["bbox"][0][0][1])
                true_x2 = (json_file["detections"][j]["bbox"][0][1][0])
                true_y2 = (json_file["detections"][j]["bbox"][0][2][1])

                true_bbox_area = (true_y2 - true_y1) * (true_x2 - true_x1)
                detected_frame_area = (detected_y2 - detected_y1) * (detected_x2 - detected_x1)

                if true_bbox_area <= detected_frame_area:
                    # top left
                    if true_x1 <= detected_x2 <= true_x2 and true_y2 >= detected_y2 >= true_y1:
                        # left
                        if true_x2 >= detected_x1 >= true_x1 and true_y1 <= detected_y2 <= true_y2:
                            intersection_area = (detected_y2 - detected_y1) * (detected_x2 - true_x1)
                        # top
                        elif true_x2 >= detected_x1 >= true_x1 and true_y1 <= detected_y2 <= true_y2:
                            intersection_area = (detected_y2 - true_y1) * (detected_x2 - detected_x1)
                        # top left
                        else:
                            intersection_area = (detected_x2 - true_x1) * (detected_y2 - true_y1)

                        if intersection_area / true_bbox_area > 0.10:
                            true_positives += 1
                        else:
                            false_positives += 1

                    # top right
                    elif true_x2 >= detected_x1 >= true_x1 and true_y1 <= detected_y2 <= true_y2:
                        # right
                        if true_x1 <= detected_x1 <= true_x2 and true_y2 >= detected_y1 >= true_y1:
                            intersection_area = (detected_y2 - detected_y1) * (true_x2 - detected_x1)
                        else:
                            intersection_area = (true_x2 - detected_x1) * (detected_y2 - true_y1)

                        if intersection_area / true_bbox_area > 0.10:
                            true_positives += 1
                        else:
                            false_positives += 1

                    # bottom left
                    elif true_x1 <= detected_x2 <= true_x2 and true_y2 >= detected_y1 >= true_y1:
                        # bottom
                        if true_x1 <= detected_x1 <= true_x2 and true_y2 >= detected_y1 >= true_y1:
                            intersection_area = (detected_x2 - detected_x1) * (true_y2 - detected_y1)
                        else:
                            intersection_area = (detected_x2 - true_x1) * (true_y2 - detected_y1)

                        if intersection_area / true_bbox_area > 0.10:
                            true_positives += 1
                        else:
                            false_positives += 1

                    # bottom right
                    elif true_x1 <= detected_x1 <= true_x2 and true_y2 >= detected_y1 >= true_y1:
                        intersection_area = (true_x2 - detected_x1) * (true_y2 - detected_y1)
                        if intersection_area / true_bbox_area > 0.10:
                            true_positives += 1
                        else:
                            false_positives += 1

                    else:
                        false_positives += 1

                # Frame > Bounding box
                else:
                    # top left
                    if detected_x2 >= true_x2 >= detected_x1 and detected_y2 >= true_y2 >= detected_y1:
                        # left
                        if detected_x2 >= true_x2 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                            intersection_area = (true_y2 - true_y1) * (true_x2 - detected_x1)
                        # top
                        elif detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y2 >= detected_y1:
                            intersection_area = (true_x2 - true_x1) * (true_y2 - detected_y1)
                        # top left
                        else:
                            intersection_area = (true_y2 - detected_y1) * (true_x2 - detected_x1)

                        if intersection_area / true_bbox_area > 0.40:
                            true_positives += 1
                        else:
                            false_positives += 1

                    # top right
                    elif detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y2 >= detected_y1:
                        # right
                        if detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                            intersection_area = (true_y2 - true_y1) * (detected_x2 - true_x1)
                        # top right
                        else:
                            intersection_area = (detected_x2 - true_x1) * (true_y2 - detected_y1)

                        if intersection_area / true_bbox_area > 0.40:
                            true_positives += 1
                        else:
                            false_positives += 1

                    # bottom left
                    elif detected_x2 >= true_x2 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                        # bottom
                        if detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                            intersection_area = (true_x2 - true_x1) * (detected_y2 - true_y1)
                        # bottom left
                        else:
                            intersection_area = (detected_y2 - true_y1) * (true_x2 - detected_x1)

                        if intersection_area / true_bbox_area > 0.40:
                            true_positives += 1
                        else:
                            false_positives += 1

                    # bottom right
                    elif detected_x2 >= true_x1 >= detected_x1 and detected_y2 >= true_y1 >= detected_y1:
                        intersection_area = (detected_x2 - true_x1) * (detected_y2 - true_y1)
                        if intersection_area / true_bbox_area > 0.40:
                            true_positives += 1
                        else:
                            false_positives += 1

                    else:
                        false_positives += 1

    if condition_counter == 0:
        false_positives += 1

    return [true_positives, false_positives]


# Helper functions to decide cropping size in width and height.
# Return factors of CROP_COUNT (number of frames that we want to crop the whole frame)
def get_factor_list(n):
    factors = [1]
    for t in range(2, (math.ceil((n / 2) + 1))):
        if n % t == 0:
            factors.append(t)
    factors.append(n)
    return factors


def factors(n):
    return iter(get_factor_list(n))


def main():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QXGA)

    # Initial declarations
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    total_positives = 0
    total_negatives = 0
    image_counter = 0

    # Loading the model
    try:
        net = tf.load(PATH_TO_TFLITE_MODEL, load_to_fb=True)
    except Exception as e:
        print(e)
        raise Exception("Failed to load model")

    directory = os.listdir(PATH_TO_DATASET)

    # Crops all photos in the dataset with respect to CROP_COUNT.
    for i in directory:
        pyb.LED(2).off()
        pyb.LED(2).on()

        window_size_h = image.Image(f"{directory}/{i}", copy_to_fb=True).height()
        window_size_w = image.Image(f"{directory}/{i}", copy_to_fb=True).width()

        # Crop Calculations
        factors = get_factor_list(CROP_COUNT)
        if len(factors) % 2 == 1:
            center_element = int(len(factors) / 2)
            crop_count_w = factors[center_element]
            crop_count_h = factors[center_element]
        else:
            center_element_1 = int(len(factors) / 2) - 1
            center_element_2 = int(len(factors) / 2)
            crop_count_w = factors[center_element_2]
            crop_count_h = factors[center_element_1]

        crop_size_width = window_size_w / crop_count_w
        crop_size_height = window_size_h / crop_count_h

        try:
            for k in range(0, (window_size_h - int(crop_size_height)) + 1, int(crop_size_height)):
                for j in range(0, (window_size_w - int(crop_size_width)) + 1, int(crop_size_width)):
                    img_grayscale = image.Image("{}/{}".format(directory, i), copy_to_fb=True).to_grayscale().crop(
                        roi=(j, k, int(crop_size_width), int(crop_size_height)))
                    img_grayscale.save(f"{PATH_TO_CROPPED_PHOTOS_TO_SAVE}/" + str(j) + "_" + str(k) + ".jpg")
                    image_counter += 1
        except Exception as e:
            print("Crop Exception" + str(e))
        pyb.LED(2).off()


        # Running the model on cropped photos and compare them with the grand truth if they exceed the limit confidence.
        for l in os.listdir(PATH_TO_CROPPED_PHOTOS_TO_SAVE):
            pyb.LED(3).off()
            pyb.LED(3).on()

            min_conf_to_accept = 0.65  # Confidences with 0.65 and higher are compared with the grand truth.

            try:
                img_name = image.Image(f"{PATH_TO_CROPPED_PHOTOS_TO_SAVE}/{l}", copy_to_fb=True).to_rgb565()
            except Exception as e1:
                print(e1)
                break

            for obj in net.classify(img_name, min_scale=1.0, scale_mul=0.5, x_overlap=0.0, y_overlap=0.0):
                x1 = l.split("_")[0]
                y1 = l.split("_")[1]
                x2 = l.split("_")[2]
                y2 = l.split("_")[3].split(".")[0]
                coords = [x1, y1, x2, y2]
                json_file_name = i.split(".")[0] + ".json"
                if obj.output()[1] > min_conf_to_accept:
                    total_positives += 1
                    true_positives += test_positive(coords, json_file_name)[0]
                    false_positives += test_positive(coords, json_file_name)[1]
                else:
                    total_negatives += 1
                    true_negatives += test_negative(coords, json_file_name)[0]
                    false_negatives += test_negative(coords, json_file_name)[1]

                print(obj.output()[1])
                print("\n\n\n")
                print("Total Positives: " + str(total_positives))
                print("True Positives: " + str(true_positives))
                print("False Positives: " + str(false_positives))
                print()
                print("Total Negatives: " + str(total_negatives))
                print("True Negatives: " + str(true_negatives))
                print("False Negatives: " + str(false_negatives))
                print("\n\n\n")

    # Calculations
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / (false_positives + false_negatives)
    F1_SCORE = 2 * (precision * recall) / (precision + recall)

    with open(PATH_TO_SAVE_INFERENCE_RESULTS, 'a') as f:
        f.write(f"Precision: {precision}\nRecall: {recall}\nAccuracy: {accuracy}\nF1 Score: {F1_SCORE}\n\n")
    

if __name__ == "__main__":
    main()
