from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy as np
import io
import wave
import six
import os

from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util
from note_seq.protobuf import music_pb2
import note_seq
from note_seq import midi_io

MAESTRO_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "maestro/train/"
)


def create_example(audio_path, _id):
    """Processes an audio file into an Example proto."""
    # print("Estamos en create example")
    wav_data = tf.gfile.Open(audio_path, "rb").read()
    example_list = list(
        audio_label_data_utils.process_record(
            wav_data=wav_data,
            ns=note_seq.NoteSequence(),
            example_id=_id,
            min_length=0,
            max_length=-1,
            allow_empty_notesequence=True,
        )
    )
    assert len(example_list) == 1
    # print(len(example_list))
    return example_list[0].SerializeToString()


def run(audio_path, _id, config_map, data_fn):
    """Create Transcription"""
    # print('CheckPoint: ', MAESTRO_CHECKPOINT_DIR)
    config = config_map["onsets_frames"]
    hparams = config.hparams
    hparams.use_cudnn = False
    hparams.batch_size = 1
    checkpoint_dir = MAESTRO_CHECKPOINT_DIR

    with tf.Graph().as_default():
        examples = tf.placeholder(tf.string, [None])

        dataset = data_fn(
            examples=examples,
            preprocess_examples=True,
            params=hparams,
            is_training=False,
            shuffle_examples=False,
            skip_n_initial_records=0,
        )
        estimator = train_util.create_estimator(
            config.model_fn, checkpoint_dir, hparams
        )
        iterator = dataset.make_initializable_iterator()
        next_record = iterator.get_next()

        with tf.Session() as sess:
            sess.run(
                [tf.initializers.global_variables(), tf.initializers.local_variables()]
            )
            sess.run(
                iterator.initializer, {examples: [create_example(audio_path, _id)]}
            )

            # print("created example")

            def transcrption_data(params):
                del params
                return tf.data.Dataset.from_tensors(sess.run(next_record))

            input_fn = infer_util.labels_to_features_wrapper(transcrption_data)
            prediction_list = list(
                estimator.predict(input_fn, yield_single_examples=False)
            )
            assert len(prediction_list) == 1

            sequence_prediction = note_seq.NoteSequence.FromString(
                prediction_list[0]["sequence_predictions"][0]
            )
            save_midi_path = (
                "/home/alan/Proyecto/Mern Auth 3/Proyecto/server/src/analisis"
            )
            midi_filename = save_midi_path + _id + ".mid"
            midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

            tf.logging.info("Transcription written to %s.", midi_filename)


def Transcribe(audio_path, _id):
    print(MAESTRO_CHECKPOINT_DIR)
    tf.disable_v2_behavior()
    run(audio_path, _id, config_map=configs.CONFIG_MAP, data_fn=data.provide_batch)
