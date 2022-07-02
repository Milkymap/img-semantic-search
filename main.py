import click 

import numpy as np 
import pickle 



from PIL import Image 
from rich.progress import track 

from os import path, getenv  
from libraries.log import logger
from libraries.strategies import *

# python main.py 
    # features_extraction

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def router_command(ctx, debug):
    ctx.ensure_object(dict)
    subcommand = ctx.invoked_subcommand 
    if subcommand is None: 
        logger.debug('use --help option in order to see the available subcommands')
    else:
        logger.debug(f'{subcommand} was called')
        path2models = getenv('MODELS')
        if path2models is None:
            logger.warning('path2models should be defined')
            logger.debug('create an env variable named MODELS')
            exit(1)
        ctx.obj['path2models'] = path2models
        
@router_command.command()
@click.option('--model_name')
@click.option('--path2images', help='path to images', type=click.Path(True))
@click.option('--extension', default='*')
@click.pass_context
def features_extraction(ctx, model_name, path2images, extension):
    path2models = ctx.obj['path2models']
    path2vectorizer = path.join(path2models, model_name)
    vectorizer = load_vectorizer(path2vectorizer)
    
    image_paths = pull_files(path2images, extension)
    nb_images = len(image_paths)
    logger.debug(f'{nb_images:04d} was found in {path2images}')
    
    accumulator = []
    for crr_path in track(image_paths, 'features extraction'):
        pil_image = Image.open(crr_path)
        fingerprint = vectorize(pil_image, vectorizer, 'cpu')
        accumulator.append(fingerprint)
    
    accumulator = np.vstack(accumulator)  # nb_imagesx512 
    serialize(accumulator, 'fingerprints.pkl', pickle)


@router_command.command()
@click.option('--request')
@click.option('--path2fingerprints')
@click.option('--model_name')
@click.option('--path2images', help='path to images', type=click.Path(True))
@click.option('--extension', default='*')
@click.pass_context
def search(ctx, request, path2fingerprints, model_name, path2images, extension):
    path2models = ctx.obj['path2models']
    path2vectorizer = path.join(path2models, model_name)
    vectorizer = load_vectorizer(path2vectorizer)
    
    image_paths = pull_files(path2images, extension)
    nb_images = len(image_paths)
    logger.debug(f'{nb_images:04d} was found in {path2images}')

    fingerprint_matrix = deserialize(path2fingerprints, pickle)
    fingerprint = vectorize(request, vectorizer, 'cpu')

    weighted_scores = scoring(fingerprint, fingerprint_matrix)
    indices = top_k(weighted_scores, 16)

    neighbor_paths = list(op.itemgetter(*indices)(image_paths))  
    response = merge_images(neighbor_paths)

    cv2.imshow('neighbors', response)
    cv2.waitKey(0)


if __name__ == '__main__':
    try:
        router_command(obj={}) 
    except Exception as e:
        logger.error(e)
