"""
Extract features. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import parse_config_file
from create_tfrecords import _convert_to_example
#from nets import nets_factory
#from preprocessing import inputs
import inputs
import model

def extract_features(tfrecords, checkpoint_path, num_iterations, feature_keys, cfg):
    """
    Extract and return the features
    """

    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()

    with graph.as_default():

        global_step = slim.get_or_create_global_step()

        
        batched_images, batched_bboxes, batched_num_bboxes, image_ids, image_names = inputs.input_nodes(
            tfrecords=tfrecords,
            max_num_bboxes = cfg.MAX_NUM_BBOXES if hasattr(cfg,'MAX_NUM_BBOXES') else 0,
            num_epochs=1,
            batch_size=cfg.BATCH_SIZE,
            num_threads=cfg.NUM_INPUT_THREADS,
            capacity=cfg.QUEUE_CAPACITY,
            min_after_dequeue=cfg.QUEUE_MIN,
            add_summaries = False,
            shuffle_batch = False,
            cfg=cfg,
        )

        arg_scope = nets_factory.arg_scopes_map[cfg.MODEL_NAME]()

        with slim.arg_scope(arg_scope):
            logits, end_points = nets_factory.networks_map[cfg.MODEL_NAME](
                inputs=batched_images,
                num_classes=cfg.NUM_CLASSES,
                is_training=False
            )

            predicted_labels = tf.argmax(end_points['Predictions'], 1)

        if 'MOVING_AVERAGE_DECAY' in cfg and cfg.MOVING_AVERAGE_DECAY > 0:
            variable_averages = tf.train.ExponentialMovingAverage(
                cfg.MOVING_AVERAGE_DECAY, global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[global_step.op.name] = global_step
        else:
            variables_to_restore = {v.name.split(':')[0]:v for v in slim.get_variables_to_restore()}

        saver = tf.train.Saver(variables_to_restore, reshape=True)

        num_batches = num_iterations
        num_items = num_batches * cfg.BATCH_SIZE

        fetches = []
        feature_stores = []
        for feature_key in feature_keys:
            feature = tf.reshape(end_points[feature_key], [cfg.BATCH_SIZE, -1])
            num_elements = feature.get_shape().as_list()[1]
            feature_stores.append(np.empty([num_items, num_elements], dtype=np.float32))
            fetches.append(feature)

        fetches.append(image_ids)
        feature_stores.append(np.empty(num_items, dtype=np.object))

        if os.path.isdir(checkpoint_path):
            checkpoint_dir = checkpoint_path
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

            if checkpoint_path is None:
                raise ValueError("Unable to find a model checkpoint in the " \
                                 "directory %s" % (checkpoint_dir,))

        tf.logging.info('Classifying records using %s' % checkpoint_path)

        coord = tf.train.Coordinator()

        sess_config = tf.ConfigProto(
                log_device_placement=cfg.SESSION_CONFIG.LOG_DEVICE_PLACEMENT,
                allow_soft_placement = True,
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
                )
            )
        sess = tf.Session(graph=graph, config=sess_config)

        with sess.as_default():

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:

                # Restore from checkpoint
                saver.restore(sess, checkpoint_path)

                print_str = ', '.join([
                  'Step: %d',
                  'Time/image (ms): %.1f'
                ])

                step = 0
                while not coord.should_stop():

                    t = time.time()
                    outputs = sess.run(fetches)
                    dt = time.time()-t

                    idx1 = cfg.BATCH_SIZE * step
                    idx2 = idx1 + cfg.BATCH_SIZE

                    for i in range(len(outputs)):
                        feature_stores[i][idx1:idx2] = outputs[i]

                    step += 1
                    print(print_str % (step, (dt / cfg.BATCH_SIZE) * 1000))

                    if num_iterations > 0 and step == num_iterations:
                        break

            except tf.errors.OutOfRangeError as e:
                pass

        coord.request_stop()
        coord.join(threads)

        feature_dict = {feature_key : feature for feature_key, feature in zip(feature_keys, feature_stores[:-1])}
        feature_dict['ids'] = feature_stores[-1]

        return feature_dict


def extract_features_to_tfrecords(tfrecords, checkpoint_path, feature_keys, cfg, output_directory):
    """
    Extract and return the features
    """

    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()

    with graph.as_default():

        global_step = slim.get_or_create_global_step()

        batched_images, batched_bboxes, batched_num_bboxes, image_ids, image_names = inputs.input_nodes(
            tfrecords=tfrecords,
            max_num_bboxes = cfg.MAX_NUM_BBOXES if hasattr(cfg,'MAX_NUM_BBOXES') else 0,
            num_epochs=1,
            batch_size=cfg.BATCH_SIZE,
            num_threads=cfg.NUM_INPUT_THREADS,
            capacity=cfg.QUEUE_CAPACITY,
            min_after_dequeue=cfg.QUEUE_MIN,
            add_summaries = False,
            shuffle_batch = False,
            cfg=cfg
        )

        with slim.arg_scope([slim.conv2d], 
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
            
            batch_norm_params = {
                'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
                'epsilon': 0.001,
                'variables_collections' : [],
                'is_training' : False
            }
            with slim.arg_scope([slim.conv2d], normalizer_params=batch_norm_params) as arg_scope:

                #arg_scope = nets_factory.arg_scopes_map[cfg.MODEL_NAME]()

                with slim.arg_scope(arg_scope):
                    features, end_points = model.inception_resnet_v2(batched_images, reuse=False, scope='InceptionResnetV2')
                    #logits, end_points = nets_factory.networks_map[cfg.MODEL_NAME](
                    #    inputs=batch_dict['inputs'],
                    #    num_classes=cfg.NUM_CLASSES,
                    #    is_training=False
                    #)
                    #
                    #predicted_labels = tf.argmax(end_points['Predictions'], 1)

                if 'MOVING_AVERAGE_DECAY' in cfg and cfg.MOVING_AVERAGE_DECAY > 0:
                    variable_averages = tf.train.ExponentialMovingAverage(
                        cfg.MOVING_AVERAGE_DECAY, global_step)
                    variables_to_restore = variable_averages.variables_to_restore(
                        slim.get_model_variables())
                    variables_to_restore[global_step.op.name] = global_step
                else:
                    variables_to_restore = {v.name.split(':')[0]:v for v in slim.get_variables_to_restore()}

        saver = tf.train.Saver(variables_to_restore, reshape=True)

        num_items, num_tf = 0, 0
        for fn in tfrecords:
            num_tf += 1
            for record in tf.python_io.tf_record_iterator(fn):
                num_items += 1
        num_iterations = int(math.ceil(float(num_items)/cfg.BATCH_SIZE))
        num_files_per_shard = cfg.BATCH_SIZE * 1000
        print("Extracting features for %d items, %d iterations, %d tfrecords, %d files/shard" % (num_items, num_iterations, num_tf, num_files_per_shard)) 
        #num_batches = num_iterations
        #num_items = num_batches * cfg.BATCH_SIZE
        
        fetches = []
        #feature_stores = []
        for feature_key in feature_keys:
            #feature = tf.reshape(end_points[feature_key], [cfg.BATCH_SIZE, -1])
            #num_elements = feature.get_shape().as_list()[1]
            #feature_stores.append(np.empty([num_items, num_elements], dtype=np.float32))
            #fetches.append(feature)
            #feature_stores.append([])
            fetches.append(end_points[feature_key])
        
        fetches.append(image_names)
        fetches.append(image_ids) #fetches.append(batch_dict['ids'])
        #feature_stores.append([])#np.empty(num_items, dtype=np.object))
        fetches.append(batched_bboxes)
        fetches.append(batched_num_bboxes)
        counter = 0

        if os.path.isdir(checkpoint_path):
            checkpoint_dir = checkpoint_path
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

            if checkpoint_path is None:
                raise ValueError("Unable to find a model checkpoint in the " \
                                 "directory %s" % (checkpoint_dir,))

        tf.logging.info('Classifying records using %s' % checkpoint_path)

        coord = tf.train.Coordinator()

        sess_config = tf.ConfigProto(
                log_device_placement=cfg.SESSION_CONFIG.LOG_DEVICE_PLACEMENT,
                allow_soft_placement = True,
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
                )
            )
        sess = tf.Session(graph=graph, config=sess_config)

        with sess.as_default():

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:

                # Restore from checkpoint
                saver.restore(sess, checkpoint_path)

                print_str = ', '.join([
                  'Step: %d',
                  'Time/image (ms): %.1f'
                ])

                step = 0
                while not coord.should_stop():

                    t = time.time()
                    outputs = sess.run(fetches)
                    dt = time.time()-t 

                    '''
                    idx1 = cfg.BATCH_SIZE * step
                    idx2 = idx1 + cfg.BATCH_SIZE
                    
                    for i in range(len(outputs)):
                        for j in range(outputs[i].shape[0]):
                            feature_stores[i].append(outputs[i][j])
                    '''
                    h, w, c = outputs[0][0].shape[0], outputs[0][0].shape[1], outputs[0][0].shape[2]
                    for j in range(cfg.BATCH_SIZE):
                        bbox = {'xmin':outputs[-2][j][:,0].tolist(),'ymin':outputs[-2][j][:,1].tolist(),'xmax':outputs[-2][j][:,2].tolist(),'ymax':outputs[-2][j][:,3].tolist()}
                        image_example = {'filename':outputs[-4][j], 'id':outputs[-3][j], 'object':{'bbox':bbox}, 'count':outputs[-1][j]}
                        example = _convert_to_example(image_example, outputs[0][j].tobytes(), outputs[0][j].shape[0], outputs[0][j].shape[1], channels=outputs[0][j].shape[2], image_format='buffer')
                        if counter % num_files_per_shard == 0:
                            output_filename = '%s-%.5d-of-%.5d' % ('features', counter/num_files_per_shard, (num_items+num_files_per_shard-1)/num_files_per_shard)
                            output_file = os.path.join(output_directory, output_filename)
                            writer = tf.python_io.TFRecordWriter(output_file)
                        writer.write(example.SerializeToString())
                        counter += 1

                    step += 1
                    print(print_str % (step, (dt / cfg.BATCH_SIZE) * 1000))

                    if num_iterations > 0 and step == num_iterations:
                        break

            except tf.errors.OutOfRangeError as e:
                pass

        coord.request_stop()
        coord.join(threads)
        return h,w,c
        '''
        feature_dict = {feature_key : np.asarray(feature) for feature_key, feature in zip(feature_keys, feature_stores[:-1])}
        feature_dict['ids'] = feature_stores[-1]

        return feature_dict
        '''

def extract_and_save(tfrecords, checkpoint_path, save_path, num_iterations, feature_keys, cfg):
    """Extract and save the features
    Args:
        tfrecords (list)
        checkpoint_path (str)
        save_dir (str)
        max_iterations (int)
        save_logits (bool)
        cfg (EasyDict)
    """

    feature_dict = extract_features(tfrecords, checkpoint_path, num_iterations, feature_keys, cfg, save_path)

    # save the results
    #np.savez(save_path, **feature_dict)


def parse_args():

    parser = argparse.ArgumentParser(description='Classify images, optionally saving the logits.')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='Paths to tfrecords.', type=str,
                        nargs='+', required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Path to a specific model to test against. If a directory, then the newest checkpoint file will be used.', type=str,
                          required=True)

    parser.add_argument('--save_path', dest='save_path',
                          help='File name path to a save the classification results.', type=str,
                          required=True)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    parser.add_argument('--batch_size', dest='batch_size',
                        help='The number of images in a batch.',
                        required=True, type=int)

    parser.add_argument('--batches', dest='batches',
                        help='Maximum number of iterations to run. Default is all records (modulo the batch size).',
                        required=True, type=int)

    parser.add_argument('--features', dest='features',
                        help='The features to extract. These are keys into the end_points dictionary returned by the model architecture.',
                        type=str, nargs='+', required=True)

    parser.add_argument('--model_name', dest='model_name',
                        help='The name of the architecture to use.',
                        required=False, type=str, default=None)



    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = parse_config_file(args.config_file)

    if args.batch_size != None:
        cfg.BATCH_SIZE = args.batch_size

    if args.model_name != None:
        cfg.MODEL_NAME = args.model_name

    extract_and_save(
        tfrecords=args.tfrecords,
        checkpoint_path=args.checkpoint_path,
        save_path = args.save_path,
        num_iterations=args.batches,
        feature_keys=args.features,
        cfg=cfg
    )

if __name__ == '__main__':
    main()
