from new_utils import create_input_files
# from create_rest import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='iu',
                       karpathy_json_path='/home/range/Data/iu_split_12_8.json',
                       image_folder='/home/range/Data/iu_xray_images/',
                       captions_per_image=1,
                       min_word_freq=5,
                       #output_folder='./peir_create_inputfile/withtag',
                       output_folder='./iu_10fold/',
                       max_len=50,
                       max_tag_len=20)
