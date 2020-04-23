# Image Classifier Script - for Command Line Application

# Inputs: 'path_to_image', 'path_to_saved_model' 

# Options: 

# --top_k, Top K most likely classes

# --category_names, Path to a JSON file mapping labels to flower names

# python predict.py /path/to/image saved_model --category_names map.json



import argparse

import numpy as np

from PIL import Image

import tensorflow_hub as hub

import json

import tensorflow as tf

import numpy as np




# predict, process_image,class_names are copy mostly from part - 1

def p_predict(image_path, model_path, top_K, category_path):


    my_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})

    img_f = Image.open(image_path)
    img_pre = np.asarray(img_f)
    img_p = process_image(img_pre)

    img_m = np.expand_dims(img_p,axis=0)

    p = my_model.predict(img_m)
    
    prediction,label = tf.math.top_k(my_model.predict(img_m), k=top_K, sorted=True)
    
    prob = prediction[0].numpy().tolist()
    
    label = label[0].numpy().tolist()
    
    print(label)
  # print(prob)
    
    if category_path != None:

        c_names = class_names(category_path)

        label_out = np.array([])
        print(label)

        for i in label:

            label_out = np.append(label_out, c_names[str(i+1)])
            print(label)
    
        label = label_out
    return prob , label



def process_image(img_t):
    img_psize = 224
    img_t = tf.cast(img_t, tf.float32)
    img_t = tf.image.resize(img_t, (img_psize, img_psize))
    img_t /= 255
    img_t = img_t.numpy()
    
    return img_t

def class_names(category_path):

    with open(category_path, 'r') as f:

        class_names = json.load(f)

    return class_names




parser = argparse.ArgumentParser(description='Given an flower image with shape (224,224,3), program predicts top K probabilities of flower class')

parser.add_argument('img_p', help='Filepath to Image')

parser.add_argument('model_p', help='Filepath to Model .H5')

parser.add_argument('-k','--top_k', type=int, help='Top K probabilities to return along with the prediction')

parser.add_argument('-c','--category_names', help='Filepath to class names JSON file')


args = parser.parse_args()


if __name__=='__main__':

    

    if args.top_k == None:

        top_k = 1

    else:

        top_k = args.top_k

    

    catg = True

    if args.category_names == None:

        catg = False

    prob, class_name = p_predict(args.img_p, args.model_p, top_k, args.category_names)

#    print(class_name[0]+1)
    
    print('\n\n')
    print('{:20}'.format('Class Name') ,'{:20}'.format('Probability'))
    
    high_prob = np.argmax(prob,axis=0)
    
    for i in range(top_k):

        if catg == False:
         
            if (i == 0):
                class_name[high_prob] = class_name[high_prob] +1
            print('{:20}'.format(str(class_name[i])), '{:3f}'.format(prob[i]))                
        else:

            print('{:20}'.format(class_name[i]), '{:3f}'.format(prob[i]))
   

    print('\n\n* The image "', args.img_p, '" belongs to class: {:10}'.format(class_name[high_prob]), ', with a probability of {:3f}'.format(prob[high_prob]))