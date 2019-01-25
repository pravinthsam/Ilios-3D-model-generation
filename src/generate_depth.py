import os
import glob
import subprocess

print('hi')

BLENDER_COMMAND = 'blender --background --python ./src/render_blender.py -- --views 10 --output_folder ./data/processed/{} {}'

for file_name in glob.glob('data/raw/**/*.obj', recursive=True):
    synset = file_name.split('/')[2]
    model_number = file_name.split('/')[3]

    print('Generating for model number : {}'.format(model_number))
    os.system(BLENDER_COMMAND.format(os.path.join(synset, model_number), file_name) + ' > /dev/null')
