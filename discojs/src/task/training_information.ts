import type { AggregatorChoice } from '../aggregator/get.js'
import type { Preprocessing } from '../dataset/data/preprocessing/index.js'
import { PreTrainedTokenizer } from '@xenova/transformers';

export interface TrainingInformation {
  // modelID: unique ID for the model
  modelID: string
  // epochs: number of epochs to run training for
  epochs: number
  // roundDuration: number of batches between each weight sharing round, e.g. if 3 then after every
  // 3 batches we share weights (in the distributed setting).
  roundDuration: number
  // validationSplit: fraction of data to keep for validation, note this only works for image data
  validationSplit: number
  // batchSize: batch size of training data
  batchSize: number
  // preprocessingFunctions: preprocessing functions such as resize and normalize
  preprocessingFunctions?: Preprocessing[]
  // dataType, 'image', 'tabular' or 'text'
  dataType: 'image' | 'tabular' | 'text'
  // inputColumns: for tabular data, the columns to be chosen as input data for the model
  inputColumns?: string[]
  // outputColumns: for tabular data, the columns to be predicted by the model
  outputColumns?: string[]
  // IMAGE_H height of image (or RESIZED_IMAGE_H if ImagePreprocessing.Resize in preprocessingFunctions)
  IMAGE_H?: number
  // IMAGE_W width of image (or RESIZED_IMAGE_W if ImagePreprocessing.Resize in preprocessingFunctions)
  IMAGE_W?: number
  // LABEL_LIST of classes, e.g. if two class of images, one with dogs and one with cats, then we would
  // define ['dogs', 'cats'].
  LABEL_LIST?: string[]
  // scheme: Distributed training scheme, i.e. Federated and Decentralized
  scheme: 'decentralized' | 'federated' | 'local'
  // noiseScale: Differential Privacy (DP): Affects the variance of the Gaussian noise added to the models / model updates.
  // Number or undefined. If undefined, then no noise will be added.
  noiseScale?: number
  // clippingRadius: Privacy (DP and Secure Aggregation):
  // Number or undefined. If undefined, then no model updates will be clipped.
  // If number, then model updates will be scaled down if their norm exceeds clippingRadius.
  clippingRadius?: number
  // decentralizedSecure: Secure Aggregation on/off:
  // Boolean. true for secure aggregation to be used, if the training scheme is decentralized, false otherwise
  decentralizedSecure?: boolean
  // maxShareValue: Secure Aggregation: maximum absolute value of a number in a randomly generated share
  // default is 100, must be a positive number, check the docs/PRIVACY.md file for more information on significance of maxShareValue selection
  // only relevant if secure aggregation is true (for either federated or decentralized learning)
  maxShareValue?: number
  // minimumReadyPeers: Decentralized Learning: minimum number of peers who must be ready to participate in aggregation before model updates are shared between clients
  // default is 3, range is [3, totalNumberOfPeersParticipating]
  minimumReadyPeers?: number
  // aggregator:  aggregator to be used by the server for federated learning, or by the peers for decentralized learning
  // default is 'average', other options include for instance 'bandit'
  aggregator?: AggregatorChoice
  // tokenizer (string | PreTrainedTokenizer). This field should be initialized with the name of a Transformers.js pre-trained tokenizer, e.g., 'Xenova/gpt2'. 
  // When the tokenizer is first called, the actual object will be initialized and loaded into this field for the subsequent tokenizations.
  tokenizer?: string | PreTrainedTokenizer
  // maxSequenceLength: the maximum length of a input string used as input to a GPT model. It is used during preprocessing to
  // truncate strings to a maximum length. The default value is tokenizer.model_max_length
  maxSequenceLength?: number
  // Tensor framework used by the model
  tensorBackend: 'tfjs' | 'gpt'
}

function isStringArray(raw: unknown): raw is string[] {
  if (!Array.isArray(raw)) {
    return false
  }
  const arr: unknown[] = raw // isArray is unsafely guarding with any[]

  return arr.every((e) => typeof e === 'string')
}

export function isTrainingInformation (raw: unknown): raw is TrainingInformation {
  if (typeof raw !== 'object' || raw === null) {
    return false
  }

  const {
    IMAGE_H,
    IMAGE_W,
    LABEL_LIST,
    aggregator,
    batchSize,
    clippingRadius,
    dataType,
    decentralizedSecure,
    epochs,
    inputColumns,
    maxShareValue,
    minimumReadyPeers,
    modelID,
    noiseScale,
    outputColumns,
    preprocessingFunctions,
    roundDuration,
    scheme,
    validationSplit,
    tokenizer,
    maxSequenceLength,
    tensorBackend
  }: Partial<Record<keyof TrainingInformation, unknown>> = raw

  if (
    typeof dataType !== 'string' ||
    typeof modelID !== 'string' ||
    typeof epochs !== 'number' ||
    typeof batchSize !== 'number' ||
    typeof roundDuration !== 'number' ||
    typeof validationSplit !== 'number' ||
    (tokenizer !== undefined && typeof tokenizer !== 'string' && !(tokenizer instanceof PreTrainedTokenizer)) ||
    (maxSequenceLength !== undefined && typeof maxSequenceLength !== 'number') ||
    (aggregator !== undefined && typeof aggregator !== 'number') ||
    (clippingRadius !== undefined && typeof clippingRadius !== 'number') ||
    (decentralizedSecure !== undefined && typeof decentralizedSecure !== 'boolean') ||
    (maxShareValue !== undefined && typeof maxShareValue !== 'number') ||
    (minimumReadyPeers !== undefined && typeof minimumReadyPeers !== 'number') ||
    (noiseScale !== undefined && typeof noiseScale !== 'number') ||
    (IMAGE_H !== undefined && typeof IMAGE_H !== 'number') ||
    (IMAGE_W !== undefined && typeof IMAGE_W !== 'number') ||
    (LABEL_LIST !== undefined && !isStringArray(LABEL_LIST)) ||
    (inputColumns !== undefined && !isStringArray(inputColumns)) ||
    (outputColumns !== undefined && !isStringArray(outputColumns)) ||
    (preprocessingFunctions !== undefined && !Array.isArray(preprocessingFunctions))
  ) {
    return false
  }

  switch (dataType) {
    case 'image': break
    case 'tabular': break
    case 'text': break
    default: return false
  }

  // interdepences on data type
  if (dataType === 'image') {
    if (typeof IMAGE_H !== 'number' || typeof IMAGE_W !== 'number') {
      return false
    }
  } else if (dataType in ['text', 'tabular']) {
    if (!(Array.isArray(inputColumns) && inputColumns.every((e) => typeof e === 'string'))) {
      return false
    }
    if (!(Array.isArray(outputColumns) && outputColumns.every((e) => typeof e === 'string'))) {
      return false
    }
  }

  switch (tensorBackend) {
    case 'tfjs': break
    case 'gpt': break
    default: return false
  }
  
  switch (scheme) {
    case 'decentralized': break
    case 'federated': break
    case 'local': break
    default: return false
  }

  const repack = {
    IMAGE_W,
    IMAGE_H,
    LABEL_LIST,
    aggregator,
    batchSize,
    clippingRadius,
    dataType,
    decentralizedSecure,
    epochs,
    inputColumns,
    maxShareValue,
    minimumReadyPeers,
    modelID,
    noiseScale,
    outputColumns,
    preprocessingFunctions,
    roundDuration,
    scheme,
    validationSplit,
    tokenizer,
    maxSequenceLength,
    tensorBackend
  }
  const _correct: TrainingInformation = repack
  const _total: Record<keyof TrainingInformation, unknown> = repack

  return true
}
