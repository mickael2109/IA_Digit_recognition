import cv2
import joblib
import numpy as np
import sympy as sp
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Charger le modèle
model = joblib.load('E:/JOBLIB/image/random_forest_3000.pkl')



#model = 'E:/JOBLIB/image/random_forest_3000.pkl'

def extract_imgs(img):
    img = ~img
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    img_data = []
    rects = []
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        rect = [x, y, w, h]
        rects.append(rect)

    bool_rect = []
    for r in rects:
        l = []
        for rec in rects:
            flag = 0
            if rec != r:
                if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (rec[1] + rec[3] + 10) and rec[1] < (r[1] + r[3] + 10):
                    flag = 1
                l.append(flag)
            else:
                l.append(0)
        bool_rect.append(l)

    dump_rect = []
    for i in range(len(cnt)):
        for j in range(len(cnt)):
            if bool_rect[i][j] == 1:
                area1 = rects[i][2] * rects[i][3]
                area2 = rects[j][2] * rects[j][3]
                if area1 == min(area1, area2):
                    dump_rect.append(rects[i])

    final_rect = [i for i in rects if i not in dump_rect]
    for r in final_rect:
        x = r[0]
        y = r[1]
        w = r[2]
        h = r[3]

        im_crop = thresh[y:y+h+10, x:x+w+10]
        im_resize = cv2.resize(im_crop, (45, 45))
        im_resize = np.reshape(im_resize, (1, 45, 45))
        img_data.append(im_resize)

    return img_data

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("L'image n'a pas pu être chargée. Vérifiez le chemin du fichier.")
    
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0
    return img.flatten()

def predict_digits(image_path, model_path):
    model = joblib.load(model_path)
    img_vector = preprocess_image(image_path)
    prediction = model.predict([img_vector])
    return int(prediction[0])

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        image_file = request.FILES['file']
        file_path = default_storage.save('temp_image.jpg', image_file)
        file_path_full = default_storage.path(file_path)
        model_path = 'E:/JOBLIB/image/random_forest_3000.pkl'

        try:
            image = cv2.imread(file_path_full, cv2.IMREAD_GRAYSCALE)
            images = extract_imgs(image)

            results = []
            for img in images:
                dataImage = []
                for element in img[0]:
                    for el in element:
                        dataImage.append(int(el))

                dataImage = dataImage[:-1]
                rep = model.predict([dataImage])
                results.append(rep[0])

            filtered_results = []
            previous_value = None
            for value in results:
                if value == 12 and previous_value == 12:
                    continue
                filtered_results.append(value)
                previous_value = value

            mapping = {
                0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: '-', 11: '+', 12: '=', 13: 'x'
            }

            equation = ''.join(mapping[val] for val in filtered_results)

            # if equation == '=' : return JsonResponse({ 'equation': equation, 'error': "Ce n'est pas une équation" })
            # if equation == '+' : return JsonResponse({ 'equation': equation, 'error': "Ce n'est pas une équation" })
            # if equation == 'x' : return JsonResponse({ 'equation': equation, 'error': "Ce n'est pas une équation" })

            if '=' in equation:
                left_side, right_side = equation.split('=')
                left_side = left_side.strip()
                right_side = right_side.strip()

                x = sp.Symbol('x')
                eq = sp.Eq(eval(left_side.replace('x', '*x').replace('-', '-1*').replace('+', '+1*')), eval(right_side))
                solution = sp.solve(eq, x)
                return JsonResponse({
                    'equation': equation,
                    'solution': str(solution)
                })

            else:
                return JsonResponse({'equation': equation, 'error': "Ce n'est pas une équation"})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)



