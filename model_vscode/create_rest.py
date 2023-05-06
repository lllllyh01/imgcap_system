import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample, shuffle
def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=50, max_tag_len=20):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    image_paths = []
    image_captions = []
    image_tags_yn=[]
    image_tags=[]
    # val_image_paths = []
    # val_image_captions = []
    # val_image_tags_yn=[]
    # val_image_tags=[]
    test_image_paths = []
    test_image_captions = []
    test_image_tags_yn=[]
    test_image_tags=[]
    train_image_len = [360, 334, 361, 368, 354, 368, 332, 355, 359, 378]
    word_freq = Counter()
    tag_freq=Counter()

    tags_list=["normal", "calcified", "granuloma", "lung", "upper", "lobe", "right", "opacity", "pulmonary", "atelectasis", "lingula", "markings", "bilateral", "interstitial", "diffuse", "prominent", "fibrosis", "mastectomy", "left", "density", "retrocardiac", "calcinosis", "blood", "vessels", "base", "bone", "diseases", "metabolic", "spine", "aorta", "tortuous", "shoulder", "degenerative", "catheters", "indwelling", "thoracic", "vertebrae", "mild", "cardiomegaly", "technical", "quality", "of", "image", "unsatisfactory", "hypoinflation", "diaphragm", "elevated", "congestion", "dislocations", "chronic", "severe", "consolidation", "costophrenic", "angle", "blunted", "surgical", "instruments", "airspace", "disease", "pleural", "effusion", "implanted", "medical", "device", "humerus", "patchy", "streaky", "pleura", "thickening", "hilum", "round", "lower", "cicatrix", "focal", "small", "hyperdistention", "sternum", "pneumothorax", "shift", "mediastinum", "nodule", "no", "indexing", "sulcus", "posterior", "obscured", "scoliosis", "bronchovascular", "granulomatous", "osteophyte", "multiple", "middle", "hernia", "hiatal", "emphysema", "atherosclerosis", "lymph", "nodes", "deformity", "anterior", "lucency", "ribs", "scattered", "lumbar", "flattened", "spondylosis", "clavicle", "irregular", "thorax", "fractures", "healed", "borderline", "kyphosis", "obstructive", "infiltrate", "heart", "failure", "edema", "moderate", "cardiac", "shadow", "enlarged", "breast", "foreign", "bodies", "spinal", "fusion", "cervical", "apex", "diaphragmatic", "eventration", "arthritis", "pneumonia", "cysts", "tuberculosis", "abdomen", "stents", "coronary", "hypertension", "hyperlucent", "hydropneumothorax", "large", "tube", "inserted", "sarcoidosis", "colonic", "interposition", "implants", "pneumoperitoneum", "sclerosis", "cholelithiasis", "epicardial", "fat", "and", "bones", "mass", "paratracheal", "artery", "supracardiac", "trachea", "carina", "hyperostosis", "idiopathic", "skeletal", "expansile", "lesions", "sinus", "ventricles", "pneumonectomy", "alveoli", "volume", "loss", "pericardial", "bronchi", "bronchitis", "reticular", "nipple", "adipose", "tissue", "subcutaneous", "neck", "blister", "azygos", "contrast", "media", "funnel", "chest", "abnormal", "aortic", "valve", "hypovolemia", "bronchiectasis", "cystic", "atria", "sutures", "acute", "aneurysm", "bullous", "cavitation", "hemopneumothorax", "mitral", "esophagus", "pectus", "carinatum", "bronchiolitis", "multilobar", "hemothorax", "cardiophrenic", "osteoporosis"]
    # tags_list = {"normal": 1, "calcified": 2, "granuloma": 3, "lung": 4, "upper": 5, "lobe": 6, "right": 7, "opacity": 8, "pulmonary": 9, "atelectasis": 10, "lingula": 11, "markings": 12, "bilateral": 13, "interstitial": 14, "diffuse": 15, "prominent": 16, "fibrosis": 17, "mastectomy": 18, "left": 19, "density": 20, "retrocardiac": 21, "calcinosis": 22, "blood": 23, "vessels": 24, "base": 25, "bone": 26, "diseases": 27, "metabolic": 28, "spine": 29, "aorta": 30, "tortuous": 31, "shoulder": 32, "degenerative": 33, "catheters": 34, "indwelling": 35, "thoracic": 36, "vertebrae": 37, "mild": 38, "cardiomegaly": 39, "technical": 40, "quality": 41, "of": 42, "image": 43, "unsatisfactory": 44, "hypoinflation": 45, "diaphragm": 46, "elevated": 47, "congestion": 48, "dislocations": 49, "chronic": 50, "severe": 51, "consolidation": 52, "costophrenic": 53, "angle": 54, "blunted": 55, "surgical": 56, "instruments": 57, "airspace": 58, "disease": 59, "pleural": 60, "effusion": 61, "implanted": 62, "medical": 63, "device": 64, "humerus": 65, "patchy": 66, "streaky": 67, "pleura": 68, "thickening": 69, "hilum": 70, "round": 71, "lower": 72, "cicatrix": 73, "focal": 74, "small": 75, "hyperdistention": 76, "sternum": 77, "pneumothorax": 78, "shift": 79, "mediastinum": 80, "nodule": 81, "no": 82, "indexing": 83, "sulcus": 84, "posterior": 85, "obscured": 86, "scoliosis": 87, "bronchovascular": 88, "granulomatous": 89, "osteophyte": 90, "multiple": 91, "middle": 92, "hernia": 93, "hiatal": 94, "emphysema": 95, "atherosclerosis": 96, "lymph": 97, "nodes": 98, "deformity": 99, "anterior": 100, "lucency": 101, "ribs": 102, "scattered": 103, "lumbar": 104, "flattened": 105, "spondylosis": 106, "clavicle": 107, "irregular": 108, "thorax": 109, "fractures": 110, "healed": 111, "borderline": 112, "kyphosis": 113, "obstructive": 114, "infiltrate": 115, "heart": 116, "failure": 117, "edema": 118, "moderate": 119, "cardiac": 120, "shadow": 121, "enlarged": 122, "breast": 123, "foreign": 124, "bodies": 125, "spinal": 126, "fusion": 127, "cervical": 128, "apex": 129, "diaphragmatic": 130, "eventration": 131, "arthritis": 132, "pneumonia": 133, "cysts": 134, "tuberculosis": 135, "abdomen": 136, "stents": 137, "coronary": 138, "hypertension": 139, "hyperlucent": 140, "hydropneumothorax": 141, "large": 142, "tube": 143, "inserted": 144, "sarcoidosis": 145, "colonic": 146, "interposition": 147, "implants": 148, "pneumoperitoneum": 149, "sclerosis": 150, "cholelithiasis": 151, "epicardial": 152, "fat": 153, "and": 154, "bones": 155, "mass": 156, "paratracheal": 157, "artery": 158, "supracardiac": 159, "trachea": 160, "carina": 161, "hyperostosis": 162, "idiopathic": 163, "skeletal": 164, "expansile": 165, "lesions": 166, "sinus": 167, "ventricles": 168, "pneumonectomy": 169, "alveoli": 170, "volume": 171, "loss": 172, "pericardial": 173, "bronchi": 174, "bronchitis": 175, "reticular": 176, "nipple": 177, "adipose": 178, "tissue": 179, "subcutaneous": 180, "neck": 181, "blister": 182, "azygos": 183, "contrast": 184, "media": 185, "funnel": 186, "chest": 187, "abnormal": 188, "aortic": 189, "valve": 190, "hypovolemia": 191, "bronchiectasis": 192, "cystic": 193, "atria": 194, "sutures": 195, "acute": 196, "aneurysm": 197, "bullous": 198, "cavitation": 199, "hemopneumothorax": 200, "mitral": 201, "esophagus": 202, "pectus": 203, "carinatum": 204, "bronchiolitis": 205, "multilobar": 206, "hemothorax": 207, "cardiophrenic": 208, "osteoporosis": 209}
    print("data['images'] len:", len(data['images']))
    print("data['images'] type:", type(data['images']))
    # tag_len = {'<10':0, '10-20':0, '20-30':0, '30-40':0, '40-50':0, '50-100':0, '>100':0}
    seed(123)
    shuffle(data['images'])
    print("first two data after shuffle:", data['images'][0]['image_id'], data['images'][1]['image_id'])
    
    word_map_file = os.path.join(output_folder, 'WORDMAP_' + dataset + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)  #1198, dict{word:No.}

    tag_map_file=os.path.join(output_folder,'TAGMAP_'+dataset+'.json')
    with open(tag_map_file,'r') as j:
        tag_map=json.load(j)

    def create_fold(impaths, imcaps, imtags, img_tags_yn, fold_id):
        # print("imtags:", imtags)
        with h5py.File(os.path.join(output_folder, str(fold_id) + '_REST_IMAGES_' + dataset + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')

            print("\nReading fold %d images and captions, storing to file...\n" % fold_id)

            captions_long_list = []
            caplens_long = []

            tags_long_list=[]
            taglens_long=[]

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)
                
                # print("imtags[i]:", imtags[i]) # [[tags_i]]
                tags=sample(imtags[i],k=1) # [[tags_i]]
                # print("tags:", tags)
                # Sanity check
                assert len(captions) == captions_per_image

                # print("captions type:", type(captions))

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (224, 224))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 224, 224)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    captions_long_list.append(enc_c)
                    caplens_long.append(c_len)
                    for j,c in enumerate(tags):
                        if len(tags) > max_tag_len:
                            tags = tags[:max_tag_len]
                        enc_c=[tag_map['<start>']] + [tag_map.get(word, tag_map['<unk>']) for word in c] + [
                            tag_map['<end>']] + [tag_map['<pad>']] * (max_tag_len - len(c))
                        c_len=len(c)+2
                        tags_long_list.append(enc_c)
                        taglens_long.append(c_len)

            # Sanity check
            # assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
            print("images len:", len(images))
            # input("Please press the Enter key to proceed")

            # Save encoded captions and their lengths to JSON files       
            with open(os.path.join(output_folder, str(fold_id) + '_REST_CAPTIONS_' + dataset + '.json'), 'w') as j:
                json.dump(captions_long_list, j)

            with open(os.path.join(output_folder, str(fold_id) + '_REST_CAPLENS_' + dataset + '.json'), 'w') as j:
                json.dump(caplens_long, j)

            with open(os.path.join(output_folder, str(fold_id) + '_REST_TAGS_' + dataset + '.json'), 'w') as j:
                json.dump(tags_long_list, j)

            with open(os.path.join(output_folder, str(fold_id) + '_REST_TAGLENS_' + dataset + '.json'), 'w') as j:
                json.dump(taglens_long, j)

            with open(os.path.join(output_folder, str(fold_id) + '_REST_TAGSYN_' + dataset + '.json'), 'w') as j:
                json.dump(img_tags_yn, j)

    
    cnt = 0
    fold_id = 0
    for img in data['images']:
        if fold_id >= 10:
            break
        if fold_id < 10 and len(image_paths) == 500-train_image_len[fold_id]: # 还有一个原先的TEST集
            create_fold(image_paths, image_captions, image_tags, image_tags_yn, fold_id) # fold_id从0开始
            fold_id += 1
            image_paths.clear()
            image_captions.clear()
            image_tags.clear()
            image_tags_yn.clear()

        captions = []
        tags=[]
        tags.append(img['tag']) # tags: [[img1_tags]]
        word_freq.update(img['caption']) # 词频统计 word_freq: {word1: freq1, word2: freq1, ...}
        tag_freq.update(img['tag'])
        tags_yn=[]
        # print("tags")
        # print(tags)
        if len(img['caption']) > max_len:
            captions.append(img['caption'])
        # captions.append(img['caption'])
        if len(captions) == 0:
            continue
        if len(tags)==0:
            continue

        for value in tags_list:
            if value in tags[0]:
                tags_yn.append(1)
            else:
                tags_yn.append(0)
        # print("tagsyn")
        # print(tags_yn)
        #for c in img['sentences']:
            # Update word frequency
            #word_freq.update(c['tokens'])

        path=os.path.join(image_folder,img['image_id']+'.png')
        # image_paths.append(path)
        # image_captions.append(captions)
        # image_tags.append(tags)
        # image_tags_yn.append(tags_yn)

        image_paths.append(path)
        image_captions.append(captions)
        image_tags.append(tags)
        image_tags_yn.append(tags_yn)
