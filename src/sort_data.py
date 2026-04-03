import os
import shutil

def sort_raw_data_by_expression(source, destination):
    os.makedirs(destination, exist_ok = True)

    valid_extensions = ('jpg', 'jpeg')

    for person in os.listdir(source):

        person_path = os.path.join(source, person)
        if not os.path.isdir(person_path): continue

        for image in os.listdir(person_path):
            if not image.endswith(valid_extensions): continue

            expression = os.path.splitext(image)[0]
            expression_folder = os.path.join(destination, expression)
            os.makedirs(expression_folder, exist_ok = True)

            shutil.copy(os.path.join(source, person_path, image),
                        os.path.join(expression_folder, f"{person}_{image}"))
        


original = "/Users/aidenyoung/pythonshi/dataproj/data/Emotional_faces"
destination = "/Users/aidenyoung/pythonshi/dataproj/data/Expressions"

sort_raw_data_by_expression(original, destination)