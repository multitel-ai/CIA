import hydra
import os
import cv2
import wget
import zipfile
import shutil

from omegaconf import DictConfig
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

from common import logger

def download_flickr(data_path: Path,
                    captions_path: str = 'Captions',
                    sentences_path: str = 'Sentences',
                    images_path: str = 'Images',
                    annotations_path: str = 'Annotations',
                    labels_path: str = 'Labels',
                    data_zip_name: str = 'flickr30k.zip'):

    image_path: Path = data_path / images_path
    annotations_path: Path = data_path / annotations_path
    caps_path: Path = data_path / captions_path
    sentences_path: Path = data_path / sentences_path
    labels_path: Path = data_path / labels_path

    dirs = data_path, image_path, annotations_path, caps_path
    logger.info(f'Attempting to create directories {[str(d) for d in dirs]}')
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    path_to_data_zip = data_path / data_zip_name

    data_url = 'http://cloud.deepilia.com/s/F83NCnFzRTJDmeY/download/flickr30k.zip'

    # Only perform the work if necessary
    if not os.path.exists(path_to_data_zip):
        logger.info(f'Downloading zip images from {data_url} to {path_to_data_zip}')
        wget.download(data_url, out=str(path_to_data_zip))
    if not len(os.listdir(image_path)):
        logger.info(f'Extracting zip images to {image_path}')
        with zipfile.ZipFile(path_to_data_zip, 'r') as zip_ref:
            zip_ref.extractall(str(data_path))

    caps_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    return image_path, annotations_path, sentences_path, caps_path, labels_path



def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to

    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence' : ' '.join(words), 'phrases' : []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index' : index,
                                             'phrase' : phrase,
                                             'phrase_id' : p_id,
                                             'phrase_type' : p_type})

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the 
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes' : {}, 'scene' : [], 'nobox' : []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info


def contains_only_one_substring(input_string, substring_list):
    count = 0
    found_substring = None

    for substring in substring_list:
        if substring in input_string:
            count += 1
            found_substring = substring

    return count == 1


def region_info_to_yolov5(image_path, region_info, class_id = 0):
    I = cv2.imread(image_path)
    image_height, image_width = I.shape[0:2]

    xmin = region_info['x_min']
    ymin = region_info['y_min']
    xmax = region_info['x_max']
    ymax = region_info['y_max']

    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    # Assemble the YOLOv5 formatted string
    yolo_str = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return yolo_str


def create_region_desc(sentence_file, annotation_file, image_id, image_file):
    sen_data = get_sentence_data(sentence_file)
    anno_data = get_annotations(annotation_file)

    image_info = {'id': image_id, 'regions': [],  'captions' : []}

    for description in sen_data:
        image_info['captions'] += [description['sentence']]

    for anno_id in anno_data['boxes'].keys():
        anno_boxes = anno_data['boxes'][anno_id]
        for anno_box in anno_boxes:
            # print()
            for description in sen_data:
                # print(description['sentence'])
                for phrase in description['phrases']:
                    # print("->>", phrase['phrase_type'], phrase)
                    if phrase['phrase_id'] == str(anno_id) \
                        and phrase['phrase_type'][0] == 'people':

                        if contains_only_one_substring(
                            phrase['phrase'].lower(), 
                            ['man', 'woman', 'lady', 'boy', 'girl', 'child', 'baby', 'person']
                        ):
                            region = region_info_to_yolov5(image_file, {
                                'x_min': anno_box[0],
                                'y_min': anno_box[1],
                                'x_max': anno_box[2],
                                'y_max': anno_box[3],
                            })

                            if region not in image_info['regions']:
                                image_info['regions'] += [region]

    return image_info



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Get all paths
    data = cfg['data']
    base_path = Path(data['base'])
    REAL_DATA_PATH = Path(base_path) / data['real']
    FLICKR_PATH = REAL_DATA_PATH / 'flickr'

    REAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
    FLICKR_PATH.mkdir(parents=True, exist_ok=True)
    
    # Download if necessary
    images_path, annotations_path, sentences_path, caps_path, labels_path = download_flickr(FLICKR_PATH)

    all_images = list(images_path.glob('*.jpg'))
    
    logger.info(f'Extracting captions and boxes info from Initial Dataset')
    for img_path in tqdm(all_images, unit='img'):
        img_path = str(img_path.absolute())
        name = img_path.split('/')[-1].split('.jpg')[0]
        # name = "11214205" # for testing
        img_file = name + '.jpg'
        txt_file = name + '.txt'
        xml_file = name + '.xml'

        image_file = images_path / img_file
        sentence_file =  sentences_path / txt_file
        annotation_file = annotations_path / xml_file
        label_file = labels_path / txt_file
        caption_file = caps_path / txt_file

        # print(image_file, sentence_file, annotation_file, label_file, caption_file)

        data = create_region_desc(sentence_file, annotation_file, name, str(image_file))

        if data['regions']:
            # os.system(f"cp flickr30k_extracted/Images/{filename} images/{filename}")
        
            # Save labels to a text file in the 'labels' folder
            with open(label_file, "w") as label_file:
                label_file.write("\n".join(data['regions']))

            # Save filtered captions to a text file in the 'captions' folder
            with open(caption_file, "w") as caption_file:
                caption_file.write("\n".join(data['captions']))

    # Prepare the data for training and validation
    real_data_images = REAL_DATA_PATH / 'images'
    real_data_labels = REAL_DATA_PATH / 'labels'
    real_data_captions = REAL_DATA_PATH / 'captions'

    real_data_images.mkdir(parents=True, exist_ok=True)
    real_data_labels.mkdir(parents=True, exist_ok=True)
    real_data_captions.mkdir(parents=True, exist_ok=True)

    TEST_NB = cfg['ml']['test_nb']
    VAL_NB = cfg['ml']['val_nb']
    TRAIN_NB = cfg['ml']['train_nb']

    logger.info(f'Moving images to {str(real_data_images)}')
    logger.info(f'Moving captions to {str(real_data_labels)}')
    logger.info(f'Moving boxes to {str(real_data_captions)}')
    logger.info(f'Using values test: {TEST_NB} and validation: {VAL_NB}')

    # move all files
    flickr_images = [label.replace('.txt', '.jpg') for label in os.listdir(labels_path)]
    print(len(flickr_images))
    length = (VAL_NB + TEST_NB + TRAIN_NB
              if (VAL_NB + TEST_NB + TRAIN_NB) < len(flickr_images)
              else flickr_images)
    flickr_images = flickr_images[:length]

    counter = 0
    print(VAL_NB + TEST_NB + TRAIN_NB, len(flickr_images))
    for file_name in tqdm(flickr_images, unit='img'):
        if counter > VAL_NB + TEST_NB + TRAIN_NB:
            break

        name =  file_name.split('.')[0]
        img_file = name + '.jpg'
        txt_file = name + '.txt'

        image = images_path / img_file
        label = labels_path / txt_file
        caption = caps_path / txt_file

        if os.path.isfile(image) and os.path.isfile(label) and os.path.isfile(caption):

            if counter < VAL_NB:
                images_dir = Path(str(real_data_images).replace(f'{os.sep}real{os.sep}', f'{os.sep}val{os.sep}'))
                labels_dir = Path(str(real_data_labels).replace(f'{os.sep}real{os.sep}', f'{os.sep}val{os.sep}'))
                captions_dir = Path(str(real_data_captions).replace(f'{os.sep}real{os.sep}', f'{os.sep}val{os.sep}'))
            elif counter < VAL_NB + TEST_NB:
                images_dir = Path(str(real_data_images).replace(f'{os.sep}real{os.sep}', f'{os.sep}test{os.sep}'))
                labels_dir = Path(str(real_data_labels).replace(f'{os.sep}real{os.sep}', f'{os.sep}test{os.sep}'))
                captions_dir = Path(str(real_data_captions).replace(f'{os.sep}real{os.sep}', f'{os.sep}test{os.sep}'))
            else:
                images_dir = real_data_images
                labels_dir = real_data_labels
                captions_dir = real_data_captions

            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            captions_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(image, images_dir / img_file)
            shutil.copy(label, labels_dir / txt_file)
            shutil.copy(caption, captions_dir / txt_file)

            counter += 1

if __name__ == "__main__":
   main()
