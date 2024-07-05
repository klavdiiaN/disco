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
exports.train = void 0;
var tf = require("@tensorflow/tfjs");
var config_js_1 = require("./config.js");
var utils_js_1 = require("./utils.js");
var evaluate_js_1 = require("./evaluate.js");
function resolveConfig(config) {
    return __assign(__assign({}, config_js_1.DEFAULT_CONFIG), config);
}
;
;
var protoClassId = (0, utils_js_1.getProtoClassIdx)({
    prototypeShape: [200, 1, 1, 128],
    numClasses: 20
});
var ppLoss = (0, utils_js_1.protoPartLoss)({
    prototypeShape: [200, 1, 1, 128]
}, protoClassId);
function train(model, ds, cfg, epochs, callbacks, evalDs) {
    return __awaiter(this, void 0, void 0, function () {
        var config, opt, _loop_1, epoch;
        var _this = this;
        var _a;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    config = resolveConfig(cfg);
                    opt = tf.train.adam(config.lr);
                    console.log('WARM');
                    _loop_1 = function (epoch) {
                        var logs, unfreeze, loss, featuresToFreeze, addOnLayersToFreeze, protsToFreeze, i;
                        return __generator(this, function (_c) {
                            switch (_c.label) {
                                case 0:
                                    logs = void 0;
                                    if (epoch === 5) {
                                        unfreeze = model.getLayer('features');
                                        unfreeze.trainable = true;
                                        console.log('JOINT');
                                    }
                                    ;
                                    ds.batch(config.batchSizeTrain).forEachAsync(function (batch) { return __awaiter(_this, void 0, void 0, function () {
                                        var _a, xs, ys, lossFn, lossTensor;
                                        return __generator(this, function (_b) {
                                            switch (_b.label) {
                                                case 0:
                                                    _a = batch, xs = _a[0], ys = _a[1];
                                                    lossFn = function () {
                                                        var output = model.predict(xs);
                                                        return ppLoss(ys, output[0]);
                                                    };
                                                    lossTensor = opt.minimize(lossFn);
                                                    return [4 /*yield*/, (lossTensor === null || lossTensor === void 0 ? void 0 : lossTensor.array())];
                                                case 1:
                                                    loss = _b.sent();
                                                    return [2 /*return*/];
                                            }
                                        });
                                    }); });
                                    return [4 /*yield*/, (0, evaluate_js_1.default)(model, evalDs, config)];
                                case 1:
                                    logs = _c.sent();
                                    console.log('epoch: ', epoch, '--- train loss: ', loss, '---balanced acc: ', logs.balanced_acc);
                                    if (!(epoch % 10 === 0)) return [3 /*break*/, 3];
                                    console.log('PUSH');
                                    featuresToFreeze = model.getLayer('features');
                                    addOnLayersToFreeze = model.getLayer('addOnLayers');
                                    protsToFreeze = model.getLayer('prototypes');
                                    featuresToFreeze.trainable, addOnLayersToFreeze.trainable, protsToFreeze.trainable = false, false, false;
                                    // final layer optimization is happening //
                                    console.log('LAST LAYER OPTIMIZATION');
                                    for (i = 0; i < 12; i++) {
                                        ds.batch(config.batchSizeTrain).forEachAsync(function (batch) { return __awaiter(_this, void 0, void 0, function () {
                                            var _a, xs, ys, lossFn, lossTensor;
                                            return __generator(this, function (_b) {
                                                switch (_b.label) {
                                                    case 0:
                                                        _a = batch, xs = _a[0], ys = _a[1];
                                                        lossFn = function () {
                                                            var output = model.predict(xs);
                                                            return ppLoss(ys, output[0]);
                                                        };
                                                        lossTensor = opt.minimize(lossFn);
                                                        return [4 /*yield*/, (lossTensor === null || lossTensor === void 0 ? void 0 : lossTensor.array())];
                                                    case 1:
                                                        loss = _b.sent();
                                                        return [2 /*return*/];
                                                }
                                            });
                                        }); });
                                        console.log('iteration: ', i, '--- train loss: ', loss);
                                    }
                                    ;
                                    return [4 /*yield*/, (0, evaluate_js_1.default)(model, evalDs, config)];
                                case 2:
                                    logs = _c.sent();
                                    featuresToFreeze.trainable, addOnLayersToFreeze.trainable, protsToFreeze.trainable = true, true, true;
                                    console.log('LAST LAYER OPTIMIZED');
                                    console.log('epoch: ', epoch, '--- train loss: ', loss, '---balanced acc: ', logs.balanced_acc);
                                    _c.label = 3;
                                case 3:
                                    ;
                                    return [4 /*yield*/, ((_a = callbacks.onEpochEnd) === null || _a === void 0 ? void 0 : _a.call(callbacks, epoch, logs))];
                                case 4:
                                    _c.sent();
                                    return [2 /*return*/];
                            }
                        });
                    };
                    epoch = 0;
                    _b.label = 1;
                case 1:
                    if (!(epoch < epochs)) return [3 /*break*/, 4];
                    return [5 /*yield**/, _loop_1(epoch)];
                case 2:
                    _b.sent();
                    _b.label = 3;
                case 3:
                    epoch++;
                    return [3 /*break*/, 1];
                case 4:
                    ;
                    opt.dispose();
                    return [2 /*return*/];
            }
        });
    });
}
exports.train = train;
