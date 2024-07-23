export interface ppnetConfig {
    lr: number
    imgSize: number
    prototypeShape: number[] // predefined [total number of prototypes, width, height, depth]
    featureShape: number[] // shape of the final convolutional output of the MobileNetV2 model
    pretrainedPath: string
    numClasses: number
    batchSizeTrain: number
    batchSizeEval: number
    batchSizePush: number
    validationSplit: number
};

export const DEFAULT_CONFIG: Required<ppnetConfig> = {
    lr: 0.001,
    imgSize: 224,
    prototypeShape: [20, 1, 1, 128],
    featureShape:[7, 7, 2048],
    //pretrainedPath: 'file://../model_cardio/model.json',
    pretrainedPath: 'https://storage.googleapis.com/deai-313515.appspot.com/models/mobileNetV2_35_alpha_2_classes/model.json',
    numClasses: 2,
    batchSizeTrain: 80,
    batchSizeEval: 127,
    batchSizePush: 75,
    validationSplit: 0.1
};