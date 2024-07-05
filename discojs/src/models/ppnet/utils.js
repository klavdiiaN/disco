"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.protoPartLoss = exports.getProtoClassIdx = void 0;
var tf = require("@tensorflow/tfjs");
var config_js_1 = require("./config.js");
function resolveConfig(config) {
    return __assign(__assign({}, config_js_1.DEFAULT_CONFIG), config);
}
;
function getProtoClassIdx(cfg) {
    //const config = resolveConfig(cfg);
    var config = Object.assign({}, cfg);
    var numClasses = config.numClasses;
    var numPrototypes = config.prototypeShape[0];
    var numPrototypePerClasses = Math.floor(numPrototypes / numClasses);
    var protoClassId = tf.zeros([numPrototypes, numClasses]);
    var protoClassIdBuffer = tf.buffer(protoClassId.shape, protoClassId.dtype, protoClassId.dataSync());
    for (var j = 0; j < numPrototypes; j++) {
        protoClassIdBuffer.set(1, j, Math.floor(j / numPrototypePerClasses));
    }
    return protoClassIdBuffer.toTensor();
}
exports.getProtoClassIdx = getProtoClassIdx;
;
function protoPartLoss(cfg, protoClassId) {
    return function (yTrue, yPred) {
        var predictions = yPred.slice(0, cfg.numClasses);
        var minDistances = yPred.slice(cfg.numClasses, cfg.prototypeShape[0]);
        var crossEntropy = tf.losses.softmaxCrossEntropy(yTrue, predictions);
        var labels = yTrue.argMax(1);
        // cluster cost
        var maxDistance = cfg.prototypeShape[1] * cfg.prototypeShape[2] * cfg.prototypeShape[3];
        var prototypesOfCorrectClass = tf.transpose(protoClassId.gather(labels, 1));
        var invertedDistances = tf.max(tf.mul(tf.sub(maxDistance, minDistances), prototypesOfCorrectClass), 1);
        var clusterCost = tf.mean(tf.sub(maxDistance, invertedDistances));
        // separation cost
        var prototypesOfWrongClass = tf.sub(tf.scalar(1), prototypesOfCorrectClass);
        var invertedDistancesNontarget = tf.max(tf.mul(tf.sub(maxDistance, minDistances), prototypesOfWrongClass), 1);
        var separationCost = tf.mean(tf.sub(maxDistance, invertedDistancesNontarget));
        return tf.addN([crossEntropy,
            tf.mul(tf.scalar(0.8), clusterCost),
            tf.mul(tf.scalar(-0.08), separationCost)
        ]);
    };
}
exports.protoPartLoss = protoPartLoss;
