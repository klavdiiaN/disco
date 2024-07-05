"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
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
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.L2Convolution_ = exports.PPNetModel = void 0;
var tf = require("@tensorflow/tfjs");
var config_js_1 = require("./config.js");
var train_js_1 = require("./train.js"); // implement
//import type { Dataset } from '../dataset/index.js'
var fs_1 = require("fs");
// SEPARATE FUNCTIONS //
function mobileNetv2(pretrainedPath) {
    return __awaiter(this, void 0, void 0, function () {
        var mobileNetv2, outputLayer, featureExtractor;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, tf.loadLayersModel(pretrainedPath)];
                case 1:
                    mobileNetv2 = _a.sent();
                    console.log(mobileNetv2.summary());
                    outputLayer = mobileNetv2.getLayer('out_relu').output;
                    featureExtractor = tf.model({ inputs: mobileNetv2.inputs, outputs: outputLayer });
                    return [2 /*return*/, featureExtractor];
            }
        });
    });
}
;
function convFeatures(cfg) {
    return __awaiter(this, void 0, void 0, function () {
        var config, inputs, x;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    config = Object.assign({
                        name: 'features',
                    }, cfg);
                    inputs = tf.input({ shape: [config.imgSize, config.imgSize, 3] });
                    console.log(inputs.shape);
                    return [4 /*yield*/, mobileNetv2(config.pretrainedPath)];
                case 1:
                    x = (_a.sent()).apply(inputs);
                    console.log(x.shape);
                    return [2 /*return*/, tf.model({ name: config.name, inputs: inputs, outputs: x })];
            }
        });
    });
}
;
function addOnLayers(features, cfg) {
    return __awaiter(this, void 0, void 0, function () {
        var config, x;
        return __generator(this, function (_a) {
            config = Object.assign({
                name: 'addOnLayers',
            }, cfg);
            x = features.output;
            x = tf.layers.conv2d({
                name: 'add_on_layer/conv2_1',
                filters: config.prototypeShape[3],
                kernelSize: 1,
                kernelInitializer: 'glorotUniform',
                activation: 'relu'
            }).apply(x);
            x = tf.layers.conv2d({
                name: 'add_on_layer/conv2d_2',
                filters: config.prototypeShape[3],
                kernelSize: 1,
                kernelInitializer: 'glorotUniform',
                activation: 'sigmoid'
            }).apply(x);
            return [2 /*return*/, tf.model({ name: config.name, inputs: features.inputs, outputs: x })];
        });
    });
}
;
// LAYERS //
var L2Convolution_ = /** @class */ (function (_super) {
    __extends(L2Convolution_, _super);
    function L2Convolution_(config) {
        var _this = _super.call(this, config) || this;
        _this.config = Object.assign({ name: 'l2_convolution' }, config);
        _this.name = 'l2_convolution';
        _this.prototypeShape = [config.prototypeShape[1], config.prototypeShape[2], config.prototypeShape[3], config.prototypeShape[0]];
        _this.featureShape = config.featureShape;
        return _this;
    }
    L2Convolution_.prototype.build = function (inputShape) {
        this.prototypeVectors = this.addWeight('proto_vec', this.prototypeShape, 'float32', tf.initializers.randomUniform({ minval: 0, maxval: 1 }));
        this.ones = this.addWeight('ones', this.prototypeShape, 'float32', tf.initializers.ones(), undefined, false);
    };
    L2Convolution_.prototype.computeOutputShape = function (inputShape) {
        return [null, this.featureShape[0], this.featureShape[1], this.prototypeShape[3]];
    };
    L2Convolution_.prototype.getConfig = function () {
        var config = _super.prototype.getConfig.call(this);
        return Object.assign({}, config, this.config);
    };
    Object.defineProperty(L2Convolution_.prototype, "protVectors", {
        get: function () {
            return this.prototypeVectors.read();
        },
        // implement //
        set: function (update) {
            this.prototypeVectors.write(update);
        },
        enumerable: false,
        configurable: true
    });
    ;
    ;
    L2Convolution_.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tf.tidy(function () {
            if (Array.isArray(inputs)) {
                inputs = inputs[0];
            }
            _this.invokeCallHook(inputs, kwargs);
            // B = batchSize, P = prototype, D = dimension, N = number
            var x2 = tf.square(inputs); // [B, 7, 7, PD]
            var x2_patch_sum = tf.conv2d(x2, _this.ones.read(), 1, 'valid'); // [B, 7, 7, PN]
            var p2 = tf.square(_this.prototypeVectors.read());
            p2 = tf.sum(p2, [0, 1, 2], false);
            p2 = tf.reshape(p2, [1, 1, -1]); // [PN]
            var xp = tf.conv2d(inputs, _this.prototypeVectors.read(), 1, 'valid');
            xp = tf.mul(xp, tf.scalar(-2)); // [B, 7, 7, PN]
            var intermediate_result = tf.add(xp, p2);
            var distances = tf.relu(tf.add(x2_patch_sum, intermediate_result));
            return distances;
        });
    };
    Object.defineProperty(L2Convolution_, "className", {
        get: function () {
            return 'L2Convolution';
        },
        enumerable: false,
        configurable: true
    });
    return L2Convolution_;
}(tf.layers.Layer));
exports.L2Convolution_ = L2Convolution_;
;
tf.serialization.registerClass(L2Convolution_);
var L2Convolution = function (config) { return new L2Convolution_(config); };
var MinDistancesPooling_ = /** @class */ (function (_super) {
    __extends(MinDistancesPooling_, _super);
    function MinDistancesPooling_(config) {
        var _this = _super.call(this, config) || this;
        _this.config = Object.assign({ name: 'min_distances' }, config);
        _this.name = 'min_distances';
        _this.kernelSize = [config.featureShape[0], config.featureShape[1]];
        _this.numPrototypes = config.prototypeShape[0];
        return _this;
    }
    MinDistancesPooling_.prototype.computeOutputShape = function (inputShape) {
        return [null, this.numPrototypes];
    };
    MinDistancesPooling_.prototype.getConfig = function () {
        var config = _super.prototype.getConfig.call(this);
        return Object.assign({}, config, this.config);
    };
    MinDistancesPooling_.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tf.tidy(function () {
            if (Array.isArray(inputs)) {
                inputs = inputs[0];
            }
            _this.invokeCallHook(inputs, kwargs);
            var distances = tf.mul(inputs, tf.scalar(-1));
            var minDistances = tf.pool(distances, _this.kernelSize, 'max', 'valid');
            minDistances = tf.mul(minDistances, tf.scalar(-1)); // [B, 1, 1, PN]
            minDistances = tf.reshape(minDistances, [-1, _this.numPrototypes]); // [B, PN]
            return minDistances;
        });
    };
    Object.defineProperty(MinDistancesPooling_, "className", {
        get: function () {
            return 'MinDistancesPooling';
        },
        enumerable: false,
        configurable: true
    });
    return MinDistancesPooling_;
}(tf.layers.Layer));
;
tf.serialization.registerClass(MinDistancesPooling_);
var MinDistancesPooling = function (config) { return new MinDistancesPooling_(config); };
var Distance2Similarity_ = /** @class */ (function (_super) {
    __extends(Distance2Similarity_, _super);
    //private prototypeActivationFunction: string;
    function Distance2Similarity_(config) {
        var _this = _super.call(this, config) || this;
        _this.config = Object.assign({ name: 'distance_to_similarity' }, config);
        _this.name = 'distance_to_similarity';
        //this.prototypeActivationFunction = config.prototypeActivationFunction;
        _this.epsilon = 1e-4;
        return _this;
    }
    Distance2Similarity_.prototype.computeOutputShape = function (inputShape) {
        return inputShape;
    };
    Distance2Similarity_.prototype.call = function (inputs, kwargs) {
        var _this = this;
        return tf.tidy(function () {
            if (Array.isArray(inputs)) {
                inputs = inputs[0];
            }
            _this.invokeCallHook(inputs, kwargs);
            return tf.log(tf.div(tf.add(inputs, tf.scalar(1)), tf.add(inputs, tf.scalar(_this.epsilon))));
        });
    };
    Object.defineProperty(Distance2Similarity_, "className", {
        get: function () {
            return 'Distance2Similarity';
        },
        enumerable: false,
        configurable: true
    });
    return Distance2Similarity_;
}(tf.layers.Layer));
;
tf.serialization.registerClass(Distance2Similarity_);
var Distance2Similarity = function (config) { return new Distance2Similarity_(config); };
function PPNet(conf) {
    return __awaiter(this, void 0, void 0, function () {
        var configDefaults, config, featureLayers, inputs, cnnFeatures, addOnFeatures, addCNN, distances, minDistances, prototype_activations, logits, combinedOutput, model, modelJson;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    configDefaults = __assign({ name: 'prototypical part network' }, config_js_1.DEFAULT_CONFIG);
                    config = Object.assign({}, configDefaults, conf);
                    return [4 /*yield*/, convFeatures(config)];
                case 1:
                    featureLayers = _a.sent();
                    inputs = tf.input({ shape: [config.imgSize, config.imgSize, 3] });
                    cnnFeatures = featureLayers.apply(inputs);
                    return [4 /*yield*/, addOnLayers(featureLayers, config)];
                case 2:
                    addOnFeatures = _a.sent();
                    addCNN = addOnFeatures.apply(cnnFeatures);
                    distances = L2Convolution(config).apply(addCNN);
                    minDistances = MinDistancesPooling(config).apply(distances);
                    prototype_activations = Distance2Similarity(config).apply(minDistances);
                    logits = tf.layers.dense({
                        name: 'logits',
                        units: config.numClasses
                    }).apply(prototype_activations);
                    combinedOutput = tf.layers.concatenate({ axis: -1 }).apply([logits, minDistances]);
                    model = tf.model({ inputs: inputs, outputs: [combinedOutput, distances] });
                    modelJson = model.toJSON();
                    (0, fs_1.writeFileSync)('ppnet.json', JSON.stringify(modelJson, null, 2));
                    return [2 /*return*/, model];
            }
        });
    });
}
;
;
// implement
var PPNetModel = /** @class */ (function (_super) {
    __extends(PPNetModel, _super);
    function PPNetModel(config) {
        var ppnet;
        var _this = _super.call(this, { inputs: [], outputs: [] }) || this; // Call super() first
        _this.config = config;
        _this.peakMemory = { value: 0 };
        PPNet(config)
            .then(function (output) {
            ppnet = output;
            var inputs = ppnet.inputs, outputs = ppnet.outputs;
            _this.inputs = inputs;
            _this.outputs = outputs;
            _this.peakMemory.value = 0;
            Object.assign(_this, ppnet); // Assign properties after super() call
        })
            .catch(function (error) {
            console.log(error);
        });
        return _this;
    }
    ;
    PPNetModel.prototype.fitDataset = function (dataset, args) {
        return __awaiter(this, void 0, void 0, function () {
            var config;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        config = __assign(__assign({}, this.config), args);
                        return [4 /*yield*/, (0, train_js_1.train)(this, dataset, config, args.epochs, args.callbacks, args.validationData)];
                    case 1:
                        _a.sent();
                        return [2 /*return*/, new tf.History()];
                }
            });
        });
    };
    return PPNetModel;
}(tf.LayersModel));
exports.PPNetModel = PPNetModel;
;
