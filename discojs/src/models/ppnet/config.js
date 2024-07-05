"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_CONFIG = void 0;
;
exports.DEFAULT_CONFIG = {
    lr: 0.001,
    imgSize: 224,
    prototypeShape: [20, 1, 1, 128],
    //prototypeActivationFunction: 'log',
    featureShape: [7, 7, 2048],
    pretrainedPath: 'https://storage.googleapis.com/deai-313515.appspot.com/models/mobileNetV2_35_alpha_2_classes/model.json',
    numClasses: 2,
    batchSizeTrain: 80,
    batchSizeEval: 127,
    batchSizePush: 75
};
