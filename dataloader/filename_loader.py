import os


def dataloader(filepath, valpath):
    # init
    up_fold = 'image_up/'
    down_fold = 'image_down/'
    disp = 'disp_up/'

    image = [
        img for img in os.listdir(filepath + up_fold) if img.find('.png') > -1
    ]
    image_val = [
        img for img in os.listdir(valpath + up_fold) if img.find('.png') > -1
    ]

    train = image[:]
    val = image_val[:]

    up_train = [filepath + up_fold + img for img in train]
    down_train = [filepath + down_fold + img for img in train]
    disp_train = [filepath + disp + img[:-4] + '.npy' for img in train]

    up_val = [valpath + up_fold + img for img in val]
    down_val = [valpath + down_fold + img for img in val]
    disp_val = [valpath + disp + img[:-4] + '.npy' for img in val]

    return up_train, down_train, disp_train, up_val, down_val, disp_val

#'''
def dataloader_2seqs(filepath1, filepath2, valpath1, valpath2):
    # Initialization
    up_fold = 'image_up/'
    down_fold = 'image_down/'
    disp = 'disp_up/'

    # Function to load images from a given path and append a suffix to filenames
    def load_images(path, suffix):
        return [img[:-4] + suffix + '.png' for img in os.listdir(path) if img.endswith('.png')]

    # Load and merge images from both file paths with suffixes
    image1 = load_images(filepath1 + up_fold, '_seq1')
    image2 = load_images(filepath2 + up_fold, '_seq2')
    train_images = image1 + image2

    image_val1 = load_images(valpath1 + up_fold, '_seq1')
    image_val2 = load_images(valpath2 + up_fold, '_seq2')
    val_images = image_val1 + image_val2

    # Function to determine the correct file path
    def determine_path(file_list, img, base1, base2):
        original_name = img.replace('_seq1', '').replace('_seq2', '')  # remove suffix to find original file
        return base1 + original_name if img in file_list else base2 + original_name

    # Create lists for training and validation data
    up_train = [determine_path(image1, img, filepath1 + up_fold, filepath2 + up_fold) for img in train_images]
    down_train = [determine_path(image1, img, filepath1 + down_fold, filepath2 + down_fold) for img in train_images]
    disp_train = [determine_path(image1, img, filepath1 + disp, filepath2 + disp)[:-4] + '.npy' for img in train_images]

    up_val = [determine_path(image_val1, img, valpath1 + up_fold, valpath2 + up_fold) for img in val_images]
    down_val = [determine_path(image_val1, img, valpath1 + down_fold, valpath2 + down_fold) for img in val_images]
    disp_val = [determine_path(image_val1, img, valpath1 + disp, valpath2 + disp)[:-4] + '.npy' for img in val_images]

    return up_train, down_train, disp_train, up_val, down_val, disp_val

def dataloader_3seqs(filepath1, filepath2, filepath3, valpath1, valpath2, valpath3):
    # Initialization
    up_fold = 'image_up/'
    down_fold = 'image_down/'
    disp = 'disp_up/'

    # Function to load images from a given path and append a suffix to filenames
    def load_images(path, suffix):
        return [img[:-4] + suffix + '.png' for img in os.listdir(path) if img.endswith('.png')]

    # Load and merge images from all file paths with suffixes
    image1 = load_images(filepath1 + up_fold, '_seq1')
    image2 = load_images(filepath2 + up_fold, '_seq2')
    image3 = load_images(filepath3 + up_fold, '_seq3')
    train_images = image1 + image2 + image3

    image_val1 = load_images(valpath1 + up_fold, '_seq1')
    image_val2 = load_images(valpath2 + up_fold, '_seq2')
    image_val3 = load_images(valpath3 + up_fold, '_seq3')
    val_images = image_val1 + image_val2 + image_val3

    print("Ici :", train_images[0])
    # Function to determine the correct file path
    def determine_path(file_lists, img, bases):
        original_name = img[:-9] + '.png'  # remove suffix to find original file
        for file_list, base in zip(file_lists, bases):
            if img in file_list:
                return base + original_name
        raise ValueError("Image {} not found in any file list".format(img))

    # Create lists for training and validation data
    up_train = [determine_path([image1, image2, image3], img, [filepath1 + up_fold, filepath2 + up_fold, filepath3 + up_fold]) for img in train_images]
    down_train = [determine_path([image1, image2, image3], img, [filepath1 + down_fold, filepath2 + down_fold, filepath3 + down_fold]) for img in train_images]
    disp_train = [determine_path([image1, image2, image3], img, [filepath1 + disp, filepath2 + disp, filepath3 + disp])[:-4] + '.npy' for img in train_images]

    up_val = [determine_path([image_val1, image_val2, image_val3], img, [valpath1 + up_fold, valpath2 + up_fold, valpath3 + up_fold]) for img in val_images]
    down_val = [determine_path([image_val1, image_val2, image_val3], img, [valpath1 + down_fold, valpath2 + down_fold, valpath3 + down_fold]) for img in val_images]
    disp_val = [determine_path([image_val1, image_val2, image_val3], img, [valpath1 + disp, valpath2 + disp, valpath3 + disp])[:-4] + '.npy' for img in val_images]

    return up_train, down_train, disp_train, up_val, down_val, disp_val

#'''