#!/usr/bin/env python
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Embeds the tokenized text contents of a TFRecord using an instance of BERT hosted on AI Platforms.

De-serializes the tokenized text contents of the target TFRecord file(s)
and performs embedding by calling the predict method of a BERT model hosted by
AI Platforms.

To use, update the following variables:
  project: A string representing the GCP project that contains the BERT model to
    be used for embedding.
  region: None, or a string representing the region hosting a model created with
    a regional endpoint.
  model: A string representing the name of the model to be used for embedding.
  model_version: The name of the model version to be used for embedding.
  tokenized_files: A list of strings indicating the path to the TFRecord file(s)
    that contain the tokenized text to be embedded.

Oputput:
  For each tokenized text record, 3 lines are printed to stdout:
    key: A unique key that can be used to translate embedded text to the original (or tokenized) text.
    pooled_embedding: An embedded representation of the entire input sequence.
    sequence_embedding: An embedded representation of each input token.
"""

import tensorflow as tf

from google.api_core.client_options import ClientOptions
from googleapiclient import discovery

# Configure these variables to match the configuration of the execution
# environment.
project = ''
region = ''
model = ''
model_version = ''
tokenized_files = [
    'data-00000-of-00001.tfrecord',
]

model_name = 'projects/{}/models/{}/versions/{}'.format(project, model,
                                                        model_version)
api_prefix = '{}-ml'.format(region) if region else 'ml'
api_endpoint = 'https://{}.googleapis.com'.format(api_prefix)
client_options = ClientOptions(api_endpoint=api_endpoint)
service = discovery.build('ml', 'v1', client_options=client_options)

# Create a serialized dataset from the list of tokenized files
serialized_dataset = tf.data.TFRecordDataset(tokenized_files)

# Describes the features for each tokenized text record stored in the example dataset.
feature_description = {
    'input_mask': tf.io.FixedLenFeature([
        128,
    ], tf.int64),
    'input_ids': tf.io.FixedLenFeature([
        128,
    ], tf.int64),
    'key': tf.io.FixedLenFeature((), tf.int64),
    'segment_ids': tf.io.FixedLenFeature([
        128,
    ], tf.int64),
}

for serialized_record in serialized_dataset:
  features = tf.io.parse_single_example(serialized_record, feature_description)

  key = str(features['key'].numpy())

  # By default, the BERT models hosted on TFHub require 3 features named
  # input_ids, input_mask and segment_ids
  instances = [{
      'input_word_ids': [int(v.numpy()) for v in list(features['input_ids'])],
      'input_mask': [int(v.numpy()) for v in list(features['input_mask'])],
      'input_type_ids': [int(v.numpy()) for v in list(features['segment_ids'])]
  }]
  response = service.projects().predict(
      name=model_name, body={
          'instances': instances
      }).execute()

  pooled_embedding = response['predictions'][0]['transformer_encoder']
  sequence_embedding = response['predictions'][0]['transformer_encoder_1']

  print(key)
  print(pooled_embedding)
  print(sequence_embedding)
